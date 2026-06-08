from pathlib import Path
import re

root = Path("CAESAR")

h = (root / "models/lbrc.h").read_text()
h = h.replace("namespace lbrc", "namespace lbrc_nvcomp_batched")
h = h.replace("} // namespace lbrc", "} // namespace lbrc_nvcomp_batched")
(root / "models/lbrc_nvcomp_batched.h").write_text(h)

cpp = (root / "models/lbrc.cpp").read_text()
cpp = cpp.replace('#include "lbrc.h"', '#include "lbrc_nvcomp_batched.h"')
cpp = cpp.replace("#include <zstd.h>\n\n", '#include <cuda_runtime.h>\n#include <nvcomp/lz4.h>\n\n')
cpp = cpp.replace("namespace lbrc", "namespace lbrc_nvcomp_batched")
cpp = cpp.replace("} // namespace lbrc", "} // namespace lbrc_nvcomp_batched")

helpers = r'''
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

'''

cpp = re.sub(
    r"void compressZstd[\s\S]*?std::vector<uint8_t> decompressZstd[\s\S]*?\n}\n\nvoid packBitsLittle",
    helpers + "\nvoid packBitsLittle",
    cpp,
    count=1,
)

cpp = cpp.replace(
    "    double sse = 0.0;\n    int64_t num = 0;\n",
    "    batchCompressBlocks(package.blocks);\n\n    double sse = 0.0;\n    int64_t num = 0;\n",
    1,
)

cpp = cpp.replace(
    "    parallelFor(package.blocks.size(), workerOptions.workers, [&](size_t i) {\n        decodeBlock(package, x0n, package.blocks[i], out);\n    });",
    "    Package decompressedPackage = package;\n    batchDecompressBlocks(decompressedPackage.blocks);\n\n    parallelFor(decompressedPackage.blocks.size(), workerOptions.workers, [&](size_t i) {\n        decodeBlock(decompressedPackage, x0n, decompressedPackage.blocks[i], out);\n    });",
    1,
)

cpp = cpp.replace("} // namespace lbrc_nvcomp_batched_batched", "} // namespace lbrc_nvcomp_batched")
(root / "models/lbrc_nvcomp_batched.cpp").write_text(cpp)

tool = (root / "tools/caesar_lbrc.cpp").read_text()
tool = tool.replace('#include "models/lbrc.h"', '#include "models/lbrc_nvcomp_batched.h"')
tool = tool.replace("caesar::lbrc::", "caesar::lbrc_nvcomp_batched::")
tool = tool.replace("CAESAR LBRC residual evaluator", "CAESAR LBRC batched nvCOMP LZ4 residual evaluator")
(root / "tools/caesar_lbrc_nvcomp_batched.cpp").write_text(tool)
