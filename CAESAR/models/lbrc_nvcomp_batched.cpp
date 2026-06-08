#include "lbrc_nvcomp_batched.h"

#include <cuda_runtime.h>
#include <nvcomp/lz4.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <fstream>
#include <future>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <thread>

namespace caesar
{
namespace lbrc_nvcomp_batched
{
namespace
{

struct Slice
{
    int64_t b;
    int64_t c;
    int64_t t0;
    int64_t t1;
    int64_t h0;
    int64_t h1;
    int64_t w0;
    int64_t w1;
};

int64_t index5(const Shape5D& s, int64_t b, int64_t c, int64_t t, int64_t h, int64_t w)
{
    return (((b * s.c + c) * s.t + t) * s.h + h) * s.w + w;
}

int64_t index3(int64_t hSize, int64_t wSize, int64_t t, int64_t h, int64_t w)
{
    return (t * hSize + h) * wSize + w;
}

std::vector<Slice> makeSlices(const Shape5D& shape, int64_t bt, int64_t bh, int64_t bw)
{
    std::vector<Slice> slices;
    for (int64_t b = 0; b < shape.b; ++b)
    {
        for (int64_t c = 0; c < shape.c; ++c)
        {
            for (int64_t t0 = 0; t0 < shape.t; t0 += bt)
            {
                for (int64_t h0 = 0; h0 < shape.h; h0 += bh)
                {
                    for (int64_t w0 = 0; w0 < shape.w; w0 += bw)
                    {
                        slices.push_back({b, c, t0, std::min(t0 + bt, shape.t),
                                          h0, std::min(h0 + bh, shape.h),
                                          w0, std::min(w0 + bw, shape.w)});
                    }
                }
            }
        }
    }
    return slices;
}


void checkCuda(cudaError_t status, const char* what)
{
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
}

void checkNvcomp(nvcompStatus_t status, const char* what)
{
    if (status != nvcompSuccess)
        throw std::runtime_error(std::string(what) + ": nvCOMP status " + std::to_string(static_cast<int>(status)));
}

// In the batched nvCOMP path, encodeBlock stores uncompressed packed bitplanes here.
// encode() compresses all of them together after all blocks are produced.
void compressZstd(const std::vector<uint8_t>& input, int, std::vector<uint8_t>& output)
{
    output = input;
}

// decode() first batch-decompresses all streams, so decodeBlock just receives packed bitplanes.
std::vector<uint8_t> decompressZstd(const std::vector<uint8_t>& input, size_t decompressedSize)
{
    if (input.size() != decompressedSize)
        throw std::runtime_error("batched nvCOMP stream has unexpected decompressed byte count");
    return input;
}

void batchCompressBlocks(std::vector<Block>& blocks)
{
    struct Ref { size_t block; size_t stream; };
    std::vector<Ref> refs;
    std::vector<size_t> inputBytes;
    size_t maxInput = 0;
    size_t totalInput = 0;

    for (size_t bi = 0; bi < blocks.size(); ++bi)
    {
        for (size_t si = 0; si < blocks[bi].streams.size(); ++si)
        {
            const size_t n = blocks[bi].streams[si].size();
            if (n == 0) continue;
            refs.push_back({bi, si});
            inputBytes.push_back(n);
            maxInput = std::max(maxInput, n);
            totalInput += n;
        }
    }

    if (refs.empty())
        return;

    const size_t chunks = refs.size();
    cudaStream_t stream{};
    checkCuda(cudaStreamCreate(&stream), "cudaStreamCreate");

    nvcompBatchedLZ4CompressOpts_t opts = nvcompBatchedLZ4CompressDefaultOpts;

    size_t tempBytes = 0;
    checkNvcomp(nvcompBatchedLZ4CompressGetTempSizeAsync(
                    chunks, maxInput, opts, &tempBytes, totalInput),
                "nvcompBatchedLZ4CompressGetTempSizeAsync");

    size_t maxOut = 0;
    checkNvcomp(nvcompBatchedLZ4CompressGetMaxOutputChunkSize(maxInput, opts, &maxOut),
                "nvcompBatchedLZ4CompressGetMaxOutputChunkSize");

    std::vector<void*> hInPtrs(chunks, nullptr), hOutPtrs(chunks, nullptr);
    void **dInPtrs = nullptr, **dOutPtrs = nullptr;
    size_t *dInBytes = nullptr, *dOutBytes = nullptr;
    nvcompStatus_t* dStatuses = nullptr;
    void* dTemp = nullptr;

    try
    {
        for (size_t i = 0; i < chunks; ++i)
        {
            const auto& s = blocks[refs[i].block].streams[refs[i].stream];
            checkCuda(cudaMalloc(&hInPtrs[i], s.size()), "cudaMalloc batched input");
            checkCuda(cudaMalloc(&hOutPtrs[i], maxOut), "cudaMalloc batched output");
            checkCuda(cudaMemcpyAsync(hInPtrs[i], s.data(), s.size(), cudaMemcpyHostToDevice, stream), "copy batched input");
        }

        checkCuda(cudaMalloc(&dInPtrs, chunks * sizeof(void*)), "cudaMalloc dInPtrs");
        checkCuda(cudaMalloc(&dOutPtrs, chunks * sizeof(void*)), "cudaMalloc dOutPtrs");
        checkCuda(cudaMalloc(&dInBytes, chunks * sizeof(size_t)), "cudaMalloc dInBytes");
        checkCuda(cudaMalloc(&dOutBytes, chunks * sizeof(size_t)), "cudaMalloc dOutBytes");
        checkCuda(cudaMalloc(&dStatuses, chunks * sizeof(nvcompStatus_t)), "cudaMalloc dStatuses");
        if (tempBytes > 0) checkCuda(cudaMalloc(&dTemp, tempBytes), "cudaMalloc dTemp");

        checkCuda(cudaMemcpyAsync(dInPtrs, hInPtrs.data(), chunks * sizeof(void*), cudaMemcpyHostToDevice, stream), "copy input ptrs");
        checkCuda(cudaMemcpyAsync(dOutPtrs, hOutPtrs.data(), chunks * sizeof(void*), cudaMemcpyHostToDevice, stream), "copy output ptrs");
        checkCuda(cudaMemcpyAsync(dInBytes, inputBytes.data(), chunks * sizeof(size_t), cudaMemcpyHostToDevice, stream), "copy input sizes");

        checkNvcomp(nvcompBatchedLZ4CompressAsync(
                        const_cast<const void**>(dInPtrs),
                        dInBytes,
                        maxInput,
                        chunks,
                        dTemp,
                        tempBytes,
                        dOutPtrs,
                        dOutBytes,
                        opts,
                        dStatuses,
                        stream),
                    "nvcompBatchedLZ4CompressAsync");

        std::vector<size_t> outputBytes(chunks);
        std::vector<nvcompStatus_t> statuses(chunks);
        checkCuda(cudaMemcpyAsync(outputBytes.data(), dOutBytes, chunks * sizeof(size_t), cudaMemcpyDeviceToHost, stream), "copy output sizes");
        checkCuda(cudaMemcpyAsync(statuses.data(), dStatuses, chunks * sizeof(nvcompStatus_t), cudaMemcpyDeviceToHost, stream), "copy statuses");
        checkCuda(cudaStreamSynchronize(stream), "sync batched compress");

        for (size_t i = 0; i < chunks; ++i)
        {
            checkNvcomp(statuses[i], "nvCOMP batched LZ4 compression chunk status");
            auto& dst = blocks[refs[i].block].streams[refs[i].stream];
            dst.resize(outputBytes[i]);
            checkCuda(cudaMemcpy(dst.data(), hOutPtrs[i], outputBytes[i], cudaMemcpyDeviceToHost), "copy compressed chunk");
        }
    }
    catch (...)
    {
        for (void* p : hInPtrs) cudaFree(p);
        for (void* p : hOutPtrs) cudaFree(p);
        cudaFree(dInPtrs); cudaFree(dOutPtrs); cudaFree(dInBytes); cudaFree(dOutBytes); cudaFree(dStatuses); cudaFree(dTemp);
        cudaStreamDestroy(stream);
        throw;
    }

    for (void* p : hInPtrs) cudaFree(p);
    for (void* p : hOutPtrs) cudaFree(p);
    cudaFree(dInPtrs); cudaFree(dOutPtrs); cudaFree(dInBytes); cudaFree(dOutBytes); cudaFree(dStatuses); cudaFree(dTemp);
    cudaStreamDestroy(stream);
}

void batchDecompressBlocks(std::vector<Block>& blocks)
{
    struct Ref { size_t block; size_t stream; size_t unpacked; };
    std::vector<Ref> refs;
    std::vector<size_t> inputBytes, outputBytes;
    size_t maxOutput = 0;
    size_t totalOutput = 0;

    for (size_t bi = 0; bi < blocks.size(); ++bi)
    {
        const size_t unpacked = static_cast<size_t>((blocks[bi].num + 7) / 8);
        for (size_t si = 0; si < blocks[bi].streams.size(); ++si)
        {
            const size_t n = blocks[bi].streams[si].size();
            if (n == 0) continue;
            refs.push_back({bi, si, unpacked});
            inputBytes.push_back(n);
            outputBytes.push_back(unpacked);
            maxOutput = std::max(maxOutput, unpacked);
            totalOutput += unpacked;
        }
    }

    if (refs.empty())
        return;

    const size_t chunks = refs.size();
    cudaStream_t stream{};
    checkCuda(cudaStreamCreate(&stream), "cudaStreamCreate");

    nvcompBatchedLZ4DecompressOpts_t opts = nvcompBatchedLZ4DecompressDefaultOpts;

    size_t tempBytes = 0;
    checkNvcomp(nvcompBatchedLZ4DecompressGetTempSizeAsync(
                    chunks, maxOutput, opts, &tempBytes, totalOutput),
                "nvcompBatchedLZ4DecompressGetTempSizeAsync");

    std::vector<void*> hInPtrs(chunks, nullptr), hOutPtrs(chunks, nullptr);
    void **dInPtrs = nullptr, **dOutPtrs = nullptr;
    size_t *dInBytes = nullptr, *dOutBufferBytes = nullptr, *dActualBytes = nullptr;
    nvcompStatus_t* dStatuses = nullptr;
    void* dTemp = nullptr;

    try
    {
        for (size_t i = 0; i < chunks; ++i)
        {
            const auto& s = blocks[refs[i].block].streams[refs[i].stream];
            checkCuda(cudaMalloc(&hInPtrs[i], s.size()), "cudaMalloc batched compressed input");
            checkCuda(cudaMalloc(&hOutPtrs[i], refs[i].unpacked), "cudaMalloc batched decompressed output");
            checkCuda(cudaMemcpyAsync(hInPtrs[i], s.data(), s.size(), cudaMemcpyHostToDevice, stream), "copy compressed chunk");
        }

        checkCuda(cudaMalloc(&dInPtrs, chunks * sizeof(void*)), "cudaMalloc dInPtrs");
        checkCuda(cudaMalloc(&dOutPtrs, chunks * sizeof(void*)), "cudaMalloc dOutPtrs");
        checkCuda(cudaMalloc(&dInBytes, chunks * sizeof(size_t)), "cudaMalloc dInBytes");
        checkCuda(cudaMalloc(&dOutBufferBytes, chunks * sizeof(size_t)), "cudaMalloc dOutBufferBytes");
        checkCuda(cudaMalloc(&dActualBytes, chunks * sizeof(size_t)), "cudaMalloc dActualBytes");
        checkCuda(cudaMalloc(&dStatuses, chunks * sizeof(nvcompStatus_t)), "cudaMalloc dStatuses");
        if (tempBytes > 0) checkCuda(cudaMalloc(&dTemp, tempBytes), "cudaMalloc dTemp");

        checkCuda(cudaMemcpyAsync(dInPtrs, hInPtrs.data(), chunks * sizeof(void*), cudaMemcpyHostToDevice, stream), "copy input ptrs");
        checkCuda(cudaMemcpyAsync(dOutPtrs, hOutPtrs.data(), chunks * sizeof(void*), cudaMemcpyHostToDevice, stream), "copy output ptrs");
        checkCuda(cudaMemcpyAsync(dInBytes, inputBytes.data(), chunks * sizeof(size_t), cudaMemcpyHostToDevice, stream), "copy input sizes");
        checkCuda(cudaMemcpyAsync(dOutBufferBytes, outputBytes.data(), chunks * sizeof(size_t), cudaMemcpyHostToDevice, stream), "copy output sizes");

        checkNvcomp(nvcompBatchedLZ4DecompressAsync(
                        const_cast<const void**>(dInPtrs),
                        dInBytes,
                        dOutBufferBytes,
                        dActualBytes,
                        chunks,
                        dTemp,
                        tempBytes,
                        dOutPtrs,
                        opts,
                        dStatuses,
                        stream),
                    "nvcompBatchedLZ4DecompressAsync");

        std::vector<size_t> actualBytes(chunks);
        std::vector<nvcompStatus_t> statuses(chunks);
        checkCuda(cudaMemcpyAsync(actualBytes.data(), dActualBytes, chunks * sizeof(size_t), cudaMemcpyDeviceToHost, stream), "copy actual sizes");
        checkCuda(cudaMemcpyAsync(statuses.data(), dStatuses, chunks * sizeof(nvcompStatus_t), cudaMemcpyDeviceToHost, stream), "copy statuses");
        checkCuda(cudaStreamSynchronize(stream), "sync batched decompress");

        for (size_t i = 0; i < chunks; ++i)
        {
            checkNvcomp(statuses[i], "nvCOMP batched LZ4 decompression chunk status");
            if (actualBytes[i] != refs[i].unpacked)
                throw std::runtime_error("nvCOMP batched LZ4 decompressed unexpected byte count");
            auto& dst = blocks[refs[i].block].streams[refs[i].stream];
            dst.resize(refs[i].unpacked);
            checkCuda(cudaMemcpy(dst.data(), hOutPtrs[i], refs[i].unpacked, cudaMemcpyDeviceToHost), "copy decompressed chunk");
        }
    }
    catch (...)
    {
        for (void* p : hInPtrs) cudaFree(p);
        for (void* p : hOutPtrs) cudaFree(p);
        cudaFree(dInPtrs); cudaFree(dOutPtrs); cudaFree(dInBytes); cudaFree(dOutBufferBytes); cudaFree(dActualBytes); cudaFree(dStatuses); cudaFree(dTemp);
        cudaStreamDestroy(stream);
        throw;
    }

    for (void* p : hInPtrs) cudaFree(p);
    for (void* p : hOutPtrs) cudaFree(p);
    cudaFree(dInPtrs); cudaFree(dOutPtrs); cudaFree(dInBytes); cudaFree(dOutBufferBytes); cudaFree(dActualBytes); cudaFree(dStatuses); cudaFree(dTemp);
    cudaStreamDestroy(stream);
}


void packBitsLittle(const std::vector<uint32_t>& values,
                    uint32_t bit,
                    std::vector<uint8_t>& packed)
{
    packed.assign((values.size() + 7) / 8, 0);
    for (size_t i = 0; i < values.size(); ++i)
    {
        if ((values[i] >> bit) & 1U)
        {
            packed[i / 8] |= static_cast<uint8_t>(1U << (i % 8));
        }
    }
}

void unpackBitsLittle(const std::vector<uint8_t>& packed,
                      uint32_t bit,
                      std::vector<uint32_t>& values)
{
    for (size_t i = 0; i < values.size(); ++i)
    {
        if ((packed[i / 8] >> (i % 8)) & 1U)
        {
            values[i] |= 1U << bit;
        }
    }
}

uint32_t zigzagEncode(int32_t value)
{
    return value >= 0 ? static_cast<uint32_t>(value) * 2U
                      : static_cast<uint32_t>(-2LL * value - 1LL);
}

int32_t zigzagDecode(uint32_t value)
{
    return (value & 1U) == 0 ? static_cast<int32_t>(value >> 1)
                             : -static_cast<int32_t>((value + 1U) >> 1);
}

std::vector<int32_t> lorenzo3D(const std::vector<int32_t>& q,
                               int64_t tSize,
                               int64_t hSize,
                               int64_t wSize)
{
    std::vector<int32_t> d(q.size(), 0);
    for (int64_t t = 0; t < tSize; ++t)
    {
        for (int64_t h = 0; h < hSize; ++h)
        {
            for (int64_t w = 0; w < wSize; ++w)
            {
                int64_t idx = index3(hSize, wSize, t, h, w);
                int64_t v = q[idx];
                if (t > 0) v -= q[index3(hSize, wSize, t - 1, h, w)];
                if (h > 0) v -= q[index3(hSize, wSize, t, h - 1, w)];
                if (w > 0) v -= q[index3(hSize, wSize, t, h, w - 1)];
                if (t > 0 && h > 0) v += q[index3(hSize, wSize, t - 1, h - 1, w)];
                if (t > 0 && w > 0) v += q[index3(hSize, wSize, t - 1, h, w - 1)];
                if (h > 0 && w > 0) v += q[index3(hSize, wSize, t, h - 1, w - 1)];
                if (t > 0 && h > 0 && w > 0)
                    v -= q[index3(hSize, wSize, t - 1, h - 1, w - 1)];
                d[idx] = static_cast<int32_t>(v);
            }
        }
    }
    return d;
}

std::vector<int32_t> inverseLorenzo3D(const std::vector<int32_t>& d,
                                      int64_t tSize,
                                      int64_t hSize,
                                      int64_t wSize)
{
    std::vector<int32_t> q(d);
    for (int64_t t = 0; t < tSize; ++t)
        for (int64_t h = 0; h < hSize; ++h)
            for (int64_t w = 1; w < wSize; ++w)
                q[index3(hSize, wSize, t, h, w)] += q[index3(hSize, wSize, t, h, w - 1)];

    for (int64_t t = 0; t < tSize; ++t)
        for (int64_t h = 1; h < hSize; ++h)
            for (int64_t w = 0; w < wSize; ++w)
                q[index3(hSize, wSize, t, h, w)] += q[index3(hSize, wSize, t, h - 1, w)];

    for (int64_t t = 1; t < tSize; ++t)
        for (int64_t h = 0; h < hSize; ++h)
            for (int64_t w = 0; w < wSize; ++w)
                q[index3(hSize, wSize, t, h, w)] += q[index3(hSize, wSize, t - 1, h, w)];

    return q;
}

double decodePathSse(const std::vector<float>& x,
                     const std::vector<float>& x0n,
                     const std::vector<float>& residual,
                     double step,
                     float mean,
                     float scale,
                     std::vector<float>& qbuf,
                     std::vector<float>& ybuf,
                     std::vector<float>& ebuf)
{
    const float step32 = static_cast<float>(step);
    double sse = 0.0;
    for (size_t i = 0; i < residual.size(); ++i)
    {
        qbuf[i] = std::nearbyint(residual[i] / step32);
        ybuf[i] = (x0n[i] + qbuf[i] * step32) * scale + mean;
        ebuf[i] = (x[i] - ybuf[i]) / scale;
        sse += static_cast<double>(ebuf[i]) * static_cast<double>(ebuf[i]);
    }
    return sse;
}

double zeroSse(const std::vector<float>& x,
               const std::vector<float>& x0n,
               float mean,
               float scale,
               std::vector<float>& ybuf,
               std::vector<float>& ebuf)
{
    double sse = 0.0;
    for (size_t i = 0; i < x.size(); ++i)
    {
        ybuf[i] = x0n[i] * scale + mean;
        ebuf[i] = (x[i] - ybuf[i]) / scale;
        sse += static_cast<double>(ebuf[i]) * static_cast<double>(ebuf[i]);
    }
    return sse;
}

Block encodeBlock(const std::vector<float>& original,
                  const std::vector<float>& x0n,
                  const std::vector<float>& residual,
                  const Shape5D& shape,
                  const Slice& sl,
                  const Options& options,
                  float mean,
                  float scale)
{
    const int64_t tSize = sl.t1 - sl.t0;
    const int64_t hSize = sl.h1 - sl.h0;
    const int64_t wSize = sl.w1 - sl.w0;
    const int64_t n = tSize * hSize * wSize;

    std::vector<float> xb(n), x0b(n), rb(n);
    int64_t p = 0;
    for (int64_t t = sl.t0; t < sl.t1; ++t)
        for (int64_t h = sl.h0; h < sl.h1; ++h)
            for (int64_t w = sl.w0; w < sl.w1; ++w, ++p)
            {
                const int64_t idx = index5(shape, sl.b, sl.c, t, h, w);
                xb[p] = original[idx];
                x0b[p] = x0n[idx];
                rb[p] = residual[idx];
            }

    std::vector<float> qbuf(n), ybuf(n), ebuf(n);
    const double targetSse = options.targetNrmse * options.targetNrmse * static_cast<double>(n);
    const double sse0 = zeroSse(xb, x0b, mean, scale, ybuf, ebuf);

    double step = 1.0;
    std::vector<int32_t> q(n, 0);
    double sse = sse0;

    if (sse0 > targetSse)
    {
        double low = 0.0;
        double high = std::max(options.targetNrmse * std::sqrt(12.0), 1e-12);
        while (decodePathSse(xb, x0b, rb, high, mean, scale, qbuf, ybuf, ebuf) <= targetSse)
        {
            low = high;
            high *= 2.0;
        }
        for (int i = 0; i < options.quantIterations; ++i)
        {
            const double mid = 0.5 * (low + high);
            if (decodePathSse(xb, x0b, rb, mid, mean, scale, qbuf, ybuf, ebuf) <= targetSse)
                low = mid;
            else
                high = mid;
        }
        step = std::max(low, 1e-12);
        sse = decodePathSse(xb, x0b, rb, step, mean, scale, qbuf, ybuf, ebuf);
        for (size_t i = 0; i < q.size(); ++i)
            q[i] = static_cast<int32_t>(std::nearbyint(rb[i] / static_cast<float>(step)));
    }

    std::vector<int32_t> d = lorenzo3D(q, tSize, hSize, wSize);
    std::vector<uint32_t> zz(d.size());
    uint32_t maxVal = 0;
    for (size_t i = 0; i < d.size(); ++i)
    {
        zz[i] = zigzagEncode(d[i]);
        maxVal = std::max(maxVal, zz[i]);
    }

    uint32_t bitCount = 1;
    while ((maxVal >> bitCount) != 0)
        ++bitCount;

    Block block;
    block.b = sl.b;
    block.c = sl.c;
    block.t0 = sl.t0;
    block.t1 = sl.t1;
    block.h0 = sl.h0;
    block.h1 = sl.h1;
    block.w0 = sl.w0;
    block.w1 = sl.w1;
    block.step = step;
    block.bitCount = bitCount;
    block.sse = sse;
    block.num = n;
    block.streams.resize(bitCount);

    std::vector<uint8_t> packed;
    for (uint32_t b = 0; b < bitCount; ++b)
    {
        packBitsLittle(zz, b, packed);
        compressZstd(packed, options.zstdLevel, block.streams[b]);
    }

    return block;
}

void decodeBlock(const Package& package,
                 const std::vector<float>& x0n,
                 const Block& block,
                 std::vector<float>& out)
{
    const int64_t tSize = block.t1 - block.t0;
    const int64_t hSize = block.h1 - block.h0;
    const int64_t wSize = block.w1 - block.w0;
    const int64_t n = tSize * hSize * wSize;
    const size_t packedBytes = static_cast<size_t>((n + 7) / 8);

    std::vector<uint32_t> zz(n, 0);
    for (uint32_t b = 0; b < block.bitCount; ++b)
    {
        std::vector<uint8_t> packed = decompressZstd(block.streams[b], packedBytes);
        unpackBitsLittle(packed, b, zz);
    }

    std::vector<int32_t> d(n);
    for (int64_t i = 0; i < n; ++i)
        d[i] = zigzagDecode(zz[i]);

    std::vector<int32_t> q = inverseLorenzo3D(d, tSize, hSize, wSize);

    int64_t p = 0;
    for (int64_t t = block.t0; t < block.t1; ++t)
        for (int64_t h = block.h0; h < block.h1; ++h)
            for (int64_t w = block.w0; w < block.w1; ++w, ++p)
            {
                const int64_t idx = index5(package.shape, block.b, block.c, t, h, w);
                const float yn = x0n[idx] + static_cast<float>(q[p]) * static_cast<float>(block.step);
                out[idx] = yn * package.scale + package.mean;
            }
}

template <typename Fn>
void parallelFor(size_t count, int workers, Fn fn)
{
    if (workers <= 1 || count <= 1)
    {
        for (size_t i = 0; i < count; ++i)
            fn(i);
        return;
    }

    const size_t nWorkers = std::min<size_t>(static_cast<size_t>(workers), count);
    std::atomic<size_t> next{0};
    std::vector<std::thread> threads;
    threads.reserve(nWorkers);
    for (size_t w = 0; w < nWorkers; ++w)
    {
        threads.emplace_back([&]() {
            while (true)
            {
                const size_t i = next.fetch_add(1);
                if (i >= count)
                    break;
                fn(i);
            }
        });
    }
    for (auto& thread : threads)
        thread.join();
}

} // namespace

std::vector<float> loadFloat32File(const std::string& path, int64_t elements)
{
    std::ifstream input(path, std::ios::binary);
    if (!input)
        throw std::runtime_error("cannot open input file: " + path);

    std::vector<float> data(static_cast<size_t>(elements));
    input.read(reinterpret_cast<char*>(data.data()),
               static_cast<std::streamsize>(data.size() * sizeof(float)));
    if (input.gcount() != static_cast<std::streamsize>(data.size() * sizeof(float)))
        throw std::runtime_error("input file is smaller than expected: " + path);
    return data;
}

void saveFloat32File(const std::string& path, const std::vector<float>& data)
{
    std::ofstream output(path, std::ios::binary);
    if (!output)
        throw std::runtime_error("cannot open output file: " + path);
    output.write(reinterpret_cast<const char*>(data.data()),
                 static_cast<std::streamsize>(data.size() * sizeof(float)));
}

Shape5D parseShape5D(const std::string& text)
{
    std::vector<int64_t> dims;
    std::stringstream ss(text);
    std::string part;
    while (std::getline(ss, part, ','))
    {
        if (!part.empty())
            dims.push_back(std::stoll(part));
    }
    if (dims.size() != 5)
        throw std::runtime_error("LBRC shape must be 5D: B,C,T,H,W");
    return {dims[0], dims[1], dims[2], dims[3], dims[4]};
}

int64_t numElements(const Shape5D& shape)
{
    return shape.b * shape.c * shape.t * shape.h * shape.w;
}

Package encode(const std::vector<float>& original,
               const std::vector<float>& baseRecon,
               const Shape5D& shape,
               const Options& options)
{
    if (original.size() != baseRecon.size())
        throw std::runtime_error("original/base reconstruction size mismatch");
    if (static_cast<int64_t>(original.size()) != numElements(shape))
        throw std::runtime_error("data size does not match LBRC shape");

    auto [minIt, maxIt] = std::minmax_element(original.begin(), original.end());
    const double sum = std::accumulate(original.begin(), original.end(), 0.0);
    const float mean = static_cast<float>(sum / std::max<size_t>(original.size(), 1));
    const float scale = static_cast<float>(*maxIt - *minIt);
    if (scale == 0.0f)
        throw std::runtime_error("zero data range");

    std::vector<float> x0n(original.size());
    std::vector<float> residual(original.size());
    for (size_t i = 0; i < original.size(); ++i)
    {
        const float xn = (original[i] - mean) / scale;
        x0n[i] = (baseRecon[i] - mean) / scale;
        residual[i] = xn - x0n[i];
    }
    Package package;
    package.shape = shape;
    package.targetNrmse = options.targetNrmse;
    package.mean = mean;
    package.scale = scale;
    package.latentBit = options.latentBit;
    package.originalBytes = static_cast<int64_t>(original.size() * sizeof(float));
    package.blockT = options.blockT;
    package.blockH = options.blockH;
    package.blockW = options.blockW;

    std::vector<Slice> slices = makeSlices(shape, options.blockT, options.blockH, options.blockW);
    package.blocks.resize(slices.size());

    Options workerOptions = options;
    if (workerOptions.workers <= 0)
        workerOptions.workers = static_cast<int>(std::thread::hardware_concurrency());

    parallelFor(slices.size(), workerOptions.workers, [&](size_t i) {
        package.blocks[i] =
            encodeBlock(original, x0n, residual, shape, slices[i], workerOptions, mean, scale);
    });

    batchCompressBlocks(package.blocks);

    double sse = 0.0;
    int64_t num = 0;
    for (const Block& block : package.blocks)
    {
        sse += block.sse;
        num += block.num;
    }
    package.encodedNrmse = std::sqrt(sse / static_cast<double>(std::max<int64_t>(num, 1)));
    return package;
}

std::vector<float> decode(const Package& package,
                          const std::vector<float>& baseRecon,
                          const Options& options)
{
    if (static_cast<int64_t>(baseRecon.size()) != numElements(package.shape))
        throw std::runtime_error("base reconstruction size does not match LBRC package");

    std::vector<float> x0n(baseRecon.size());
    for (size_t i = 0; i < baseRecon.size(); ++i)
        x0n[i] = (baseRecon[i] - package.mean) / package.scale;

    std::vector<float> out(baseRecon.size());

    Options workerOptions = options;
    if (workerOptions.workers <= 0)
        workerOptions.workers = static_cast<int>(std::thread::hardware_concurrency());

    Package decompressedPackage = package;
    batchDecompressBlocks(decompressedPackage.blocks);

    parallelFor(decompressedPackage.blocks.size(), workerOptions.workers, [&](size_t i) {
        decodeBlock(decompressedPackage, x0n, decompressedPackage.blocks[i], out);
    });
    return out;
}

EvalResult evaluate(const std::vector<float>& original,
                    const std::vector<float>& baseRecon,
                    const Shape5D& shape,
                    const Options& options)
{
    Package package = encode(original, baseRecon, shape, options);
    std::vector<float> recon = decode(package, baseRecon, options);

    EvalResult result;
    result.finalNrmse = nrmse(original, recon, package.scale);
    result.encodedNrmse = package.encodedNrmse;
    result.correctionBytes = serializedSize(package);
    result.latentBytes = static_cast<size_t>(std::max<int64_t>(options.latentBit, 0)) / 8;
    const double totalBytes =
        static_cast<double>(result.correctionBytes) + static_cast<double>(result.latentBytes);
    result.cr = totalBytes > 0.0 ? static_cast<double>(original.size() * sizeof(float)) / totalBytes : 0.0;
    result.blocks = package.blocks.size();
    return result;
}

size_t serializedSize(const Package& package)
{
    size_t bytes = 0;
    bytes += sizeof(package.shape);
    bytes += sizeof(package.targetNrmse);
    bytes += sizeof(package.mean);
    bytes += sizeof(package.scale);
    bytes += sizeof(package.latentBit);
    bytes += sizeof(package.originalBytes);
    bytes += sizeof(package.blockT) + sizeof(package.blockH) + sizeof(package.blockW);
    bytes += sizeof(package.encodedNrmse);
    bytes += sizeof(uint64_t);
    for (const Block& block : package.blocks)
    {
        bytes += sizeof(block.b) * 8;
        bytes += sizeof(block.step);
        bytes += sizeof(block.bitCount);
        bytes += sizeof(block.sse);
        bytes += sizeof(block.num);
        bytes += sizeof(uint64_t);
        for (const auto& stream : block.streams)
            bytes += sizeof(uint64_t) + stream.size();
    }
    return bytes;
}

double nrmse(const std::vector<float>& a,
             const std::vector<float>& b,
             double scale)
{
    if (a.size() != b.size())
        throw std::runtime_error("nrmse input size mismatch");
    double mse = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
    {
        const double e = (static_cast<double>(a[i]) - static_cast<double>(b[i])) / scale;
        mse += e * e;
    }
    return std::sqrt(mse / static_cast<double>(std::max<size_t>(a.size(), 1)));
}

} // namespace lbrc_nvcomp_batched_nvcomp_batched
} // namespace caesar
