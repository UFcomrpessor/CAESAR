from pathlib import Path
import re

root = Path("CAESAR")

h = (root / "models/lbrc.h").read_text()
h = h.replace("namespace lbrc", "namespace lbrc_nvcomp")
h = h.replace("} // namespace lbrc", "} // namespace lbrc_nvcomp")
(root / "models/lbrc_nvcomp.h").write_text(h)

cpp = (root / "models/lbrc.cpp").read_text()
cpp = cpp.replace('#include "lbrc.h"', '#include "lbrc_nvcomp.h"')
cpp = cpp.replace("#include <zstd.h>\n\n", '#include <cuda_runtime.h>\n#include <nvcomp/lz4.h>\n\n')
cpp = cpp.replace("namespace lbrc", "namespace lbrc_nvcomp")
cpp = cpp.replace("} // namespace lbrc", "} // namespace lbrc_nvcomp")

replacement = r'''
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

void compressZstd(const std::vector<uint8_t>& input, int, std::vector<uint8_t>& output)
{
    if (input.empty())
    {
        output.clear();
        return;
    }

    cudaStream_t stream{};
    checkCuda(cudaStreamCreate(&stream), "cudaStreamCreate");

    nvcompBatchedLZ4CompressOpts_t opts = nvcompBatchedLZ4CompressDefaultOpts;

    size_t tempBytes = 0;
    checkNvcomp(nvcompBatchedLZ4CompressGetTempSizeAsync(
                    1, input.size(), opts, &tempBytes, input.size()),
                "nvcompBatchedLZ4CompressGetTempSizeAsync");

    size_t maxOutBytes = 0;
    checkNvcomp(nvcompBatchedLZ4CompressGetMaxOutputChunkSize(
                    input.size(), opts, &maxOutBytes),
                "nvcompBatchedLZ4CompressGetMaxOutputChunkSize");

    void* dInput = nullptr;
    void* dOutput = nullptr;
    void* dTemp = nullptr;
    void** dInputPtrs = nullptr;
    void** dOutputPtrs = nullptr;
    size_t* dInputBytes = nullptr;
    size_t* dOutputBytes = nullptr;
    nvcompStatus_t* dStatuses = nullptr;

    try
    {
        checkCuda(cudaMalloc(&dInput, input.size()), "cudaMalloc dInput");
        checkCuda(cudaMalloc(&dOutput, maxOutBytes), "cudaMalloc dOutput");
        if (tempBytes > 0) checkCuda(cudaMalloc(&dTemp, tempBytes), "cudaMalloc dTemp");
        checkCuda(cudaMalloc(&dInputPtrs, sizeof(void*)), "cudaMalloc dInputPtrs");
        checkCuda(cudaMalloc(&dOutputPtrs, sizeof(void*)), "cudaMalloc dOutputPtrs");
        checkCuda(cudaMalloc(&dInputBytes, sizeof(size_t)), "cudaMalloc dInputBytes");
        checkCuda(cudaMalloc(&dOutputBytes, sizeof(size_t)), "cudaMalloc dOutputBytes");
        checkCuda(cudaMalloc(&dStatuses, sizeof(nvcompStatus_t)), "cudaMalloc dStatuses");

        void* hInputPtr = dInput;
        void* hOutputPtr = dOutput;
        size_t hInputBytes = input.size();

        checkCuda(cudaMemcpyAsync(dInput, input.data(), input.size(), cudaMemcpyHostToDevice, stream), "copy input");
        checkCuda(cudaMemcpyAsync(dInputPtrs, &hInputPtr, sizeof(void*), cudaMemcpyHostToDevice, stream), "copy input ptr");
        checkCuda(cudaMemcpyAsync(dOutputPtrs, &hOutputPtr, sizeof(void*), cudaMemcpyHostToDevice, stream), "copy output ptr");
        checkCuda(cudaMemcpyAsync(dInputBytes, &hInputBytes, sizeof(size_t), cudaMemcpyHostToDevice, stream), "copy input bytes");

        checkNvcomp(nvcompBatchedLZ4CompressAsync(
                        const_cast<const void**>(dInputPtrs),
                        dInputBytes,
                        input.size(),
                        1,
                        dTemp,
                        tempBytes,
                        dOutputPtrs,
                        dOutputBytes,
                        opts,
                        dStatuses,
                        stream),
                    "nvcompBatchedLZ4CompressAsync");

        size_t hOutputBytes = 0;
        nvcompStatus_t hStatus{};
        checkCuda(cudaMemcpyAsync(&hOutputBytes, dOutputBytes, sizeof(size_t), cudaMemcpyDeviceToHost, stream), "copy output bytes");
        checkCuda(cudaMemcpyAsync(&hStatus, dStatuses, sizeof(nvcompStatus_t), cudaMemcpyDeviceToHost, stream), "copy status");
        checkCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize compress");
        checkNvcomp(hStatus, "nvcomp LZ4 chunk compress status");

        output.resize(hOutputBytes);
        checkCuda(cudaMemcpy(output.data(), dOutput, hOutputBytes, cudaMemcpyDeviceToHost), "copy compressed output");
    }
    catch (...)
    {
        cudaFree(dInput); cudaFree(dOutput); cudaFree(dTemp);
        cudaFree(dInputPtrs); cudaFree(dOutputPtrs);
        cudaFree(dInputBytes); cudaFree(dOutputBytes); cudaFree(dStatuses);
        cudaStreamDestroy(stream);
        throw;
    }

    cudaFree(dInput); cudaFree(dOutput); cudaFree(dTemp);
    cudaFree(dInputPtrs); cudaFree(dOutputPtrs);
    cudaFree(dInputBytes); cudaFree(dOutputBytes); cudaFree(dStatuses);
    cudaStreamDestroy(stream);
}

std::vector<uint8_t> decompressZstd(const std::vector<uint8_t>& input, size_t decompressedSize)
{
    if (decompressedSize == 0)
        return {};

    cudaStream_t stream{};
    checkCuda(cudaStreamCreate(&stream), "cudaStreamCreate");

    nvcompBatchedLZ4DecompressOpts_t opts = nvcompBatchedLZ4DecompressDefaultOpts;

    size_t tempBytes = 0;
    checkNvcomp(nvcompBatchedLZ4DecompressGetTempSizeAsync(
                    1, decompressedSize, opts, &tempBytes, decompressedSize),
                "nvcompBatchedLZ4DecompressGetTempSizeAsync");

    void* dInput = nullptr;
    void* dOutput = nullptr;
    void* dTemp = nullptr;
    void** dInputPtrs = nullptr;
    void** dOutputPtrs = nullptr;
    size_t* dInputBytes = nullptr;
    size_t* dOutputBufferBytes = nullptr;
    size_t* dActualBytes = nullptr;
    nvcompStatus_t* dStatuses = nullptr;

    std::vector<uint8_t> output(decompressedSize);

    try
    {
        checkCuda(cudaMalloc(&dInput, input.size()), "cudaMalloc dInput");
        checkCuda(cudaMalloc(&dOutput, decompressedSize), "cudaMalloc dOutput");
        if (tempBytes > 0) checkCuda(cudaMalloc(&dTemp, tempBytes), "cudaMalloc dTemp");
        checkCuda(cudaMalloc(&dInputPtrs, sizeof(void*)), "cudaMalloc dInputPtrs");
        checkCuda(cudaMalloc(&dOutputPtrs, sizeof(void*)), "cudaMalloc dOutputPtrs");
        checkCuda(cudaMalloc(&dInputBytes, sizeof(size_t)), "cudaMalloc dInputBytes");
        checkCuda(cudaMalloc(&dOutputBufferBytes, sizeof(size_t)), "cudaMalloc dOutputBufferBytes");
        checkCuda(cudaMalloc(&dActualBytes, sizeof(size_t)), "cudaMalloc dActualBytes");
        checkCuda(cudaMalloc(&dStatuses, sizeof(nvcompStatus_t)), "cudaMalloc dStatuses");

        void* hInputPtr = dInput;
        void* hOutputPtr = dOutput;
        size_t hInputBytes = input.size();
        size_t hOutputBufferBytes = decompressedSize;

        checkCuda(cudaMemcpyAsync(dInput, input.data(), input.size(), cudaMemcpyHostToDevice, stream), "copy compressed input");
        checkCuda(cudaMemcpyAsync(dInputPtrs, &hInputPtr, sizeof(void*), cudaMemcpyHostToDevice, stream), "copy input ptr");
        checkCuda(cudaMemcpyAsync(dOutputPtrs, &hOutputPtr, sizeof(void*), cudaMemcpyHostToDevice, stream), "copy output ptr");
        checkCuda(cudaMemcpyAsync(dInputBytes, &hInputBytes, sizeof(size_t), cudaMemcpyHostToDevice, stream), "copy input bytes");
        checkCuda(cudaMemcpyAsync(dOutputBufferBytes, &hOutputBufferBytes, sizeof(size_t), cudaMemcpyHostToDevice, stream), "copy output bytes");

        checkNvcomp(nvcompBatchedLZ4DecompressAsync(
                        const_cast<const void**>(dInputPtrs),
                        dInputBytes,
                        dOutputBufferBytes,
                        dActualBytes,
                        1,
                        dTemp,
                        tempBytes,
                        dOutputPtrs,
                        opts,
                        dStatuses,
                        stream),
                    "nvcompBatchedLZ4DecompressAsync");

        size_t hActualBytes = 0;
        nvcompStatus_t hStatus{};
        checkCuda(cudaMemcpyAsync(&hActualBytes, dActualBytes, sizeof(size_t), cudaMemcpyDeviceToHost, stream), "copy actual bytes");
        checkCuda(cudaMemcpyAsync(&hStatus, dStatuses, sizeof(nvcompStatus_t), cudaMemcpyDeviceToHost, stream), "copy status");
        checkCuda(cudaMemcpyAsync(output.data(), dOutput, decompressedSize, cudaMemcpyDeviceToHost, stream), "copy decompressed output");
        checkCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize decompress");
        checkNvcomp(hStatus, "nvcomp LZ4 chunk decompress status");
        if (hActualBytes != decompressedSize)
            throw std::runtime_error("nvCOMP LZ4 decompressed unexpected byte count");
    }
    catch (...)
    {
        cudaFree(dInput); cudaFree(dOutput); cudaFree(dTemp);
        cudaFree(dInputPtrs); cudaFree(dOutputPtrs);
        cudaFree(dInputBytes); cudaFree(dOutputBufferBytes); cudaFree(dActualBytes); cudaFree(dStatuses);
        cudaStreamDestroy(stream);
        throw;
    }

    cudaFree(dInput); cudaFree(dOutput); cudaFree(dTemp);
    cudaFree(dInputPtrs); cudaFree(dOutputPtrs);
    cudaFree(dInputBytes); cudaFree(dOutputBufferBytes); cudaFree(dActualBytes); cudaFree(dStatuses);
    cudaStreamDestroy(stream);

    return output;
}

'''

cpp = re.sub(
    r"void compressZstd[\s\S]*?std::vector<uint8_t> decompressZstd[\s\S]*?\n}\n\nvoid packBitsLittle",
    replacement + "\nvoid packBitsLittle",
    cpp,
    count=1,
)
(root / "models/lbrc_nvcomp.cpp").write_text(cpp)

tool = (root / "tools/caesar_lbrc.cpp").read_text()
tool = tool.replace('#include "models/lbrc.h"', '#include "models/lbrc_nvcomp.h"')
tool = tool.replace("caesar::lbrc::", "caesar::lbrc_nvcomp::")
tool = tool.replace("CAESAR LBRC residual evaluator", "CAESAR LBRC nvCOMP LZ4 residual evaluator")
(root / "tools/caesar_lbrc_nvcomp.cpp").write_text(tool)
