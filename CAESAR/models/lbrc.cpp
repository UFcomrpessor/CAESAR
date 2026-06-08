#include "lbrc.h"

#include <zstd.h>

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
namespace lbrc
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

void compressZstd(const std::vector<uint8_t>& input, int level, std::vector<uint8_t>& output)
{
    const size_t bound = ZSTD_compressBound(input.size());
    output.resize(bound);
    const size_t written =
        ZSTD_compress(output.data(), output.size(), input.data(), input.size(), level);
    if (ZSTD_isError(written))
    {
        throw std::runtime_error(std::string("ZSTD_compress failed: ") + ZSTD_getErrorName(written));
    }
    output.resize(written);
}

std::vector<uint8_t> decompressZstd(const std::vector<uint8_t>& input, size_t decompressedSize)
{
    std::vector<uint8_t> output(decompressedSize);
    const size_t written =
        ZSTD_decompress(output.data(), output.size(), input.data(), input.size());
    if (ZSTD_isError(written))
    {
        throw std::runtime_error(std::string("ZSTD_decompress failed: ") + ZSTD_getErrorName(written));
    }
    if (written != decompressedSize)
    {
        throw std::runtime_error("ZSTD_decompress produced unexpected byte count");
    }
    return output;
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

    parallelFor(package.blocks.size(), workerOptions.workers, [&](size_t i) {
        decodeBlock(package, x0n, package.blocks[i], out);
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

} // namespace lbrc
} // namespace caesar
