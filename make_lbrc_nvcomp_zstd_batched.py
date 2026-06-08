from pathlib import Path

root = Path("CAESAR")

h = (root / "models/lbrc_nvcomp_batched.h").read_text()
h = h.replace("namespace lbrc_nvcomp_batched", "namespace lbrc_nvcomp_zstd_batched")
h = h.replace("} // namespace lbrc_nvcomp_batched", "} // namespace lbrc_nvcomp_zstd_batched")
(root / "models/lbrc_nvcomp_zstd_batched.h").write_text(h)

cpp = (root / "models/lbrc_nvcomp_batched.cpp").read_text()
cpp = cpp.replace('#include "lbrc_nvcomp_batched.h"', '#include "lbrc_nvcomp_zstd_batched.h"')
cpp = cpp.replace("#include <nvcomp/lz4.h>", "#include <nvcomp/zstd.h>")
cpp = cpp.replace("namespace lbrc_nvcomp_batched", "namespace lbrc_nvcomp_zstd_batched")
cpp = cpp.replace("} // namespace lbrc_nvcomp_batched", "} // namespace lbrc_nvcomp_zstd_batched")

repls = {
    "nvcompBatchedLZ4CompressOpts_t": "nvcompBatchedZstdCompressOpts_t",
    "nvcompBatchedLZ4DecompressOpts_t": "nvcompBatchedZstdDecompressOpts_t",
    "nvcompBatchedLZ4CompressDefaultOpts": "nvcompBatchedZstdCompressDefaultOpts",
    "nvcompBatchedLZ4DecompressDefaultOpts": "nvcompBatchedZstdDecompressDefaultOpts",
    "nvcompBatchedLZ4CompressGetTempSizeAsync": "nvcompBatchedZstdCompressGetTempSizeAsync",
    "nvcompBatchedLZ4CompressGetMaxOutputChunkSize": "nvcompBatchedZstdCompressGetMaxOutputChunkSize",
    "nvcompBatchedLZ4CompressAsync": "nvcompBatchedZstdCompressAsync",
    "nvcompBatchedLZ4DecompressGetTempSizeAsync": "nvcompBatchedZstdDecompressGetTempSizeAsync",
    "nvcompBatchedLZ4DecompressAsync": "nvcompBatchedZstdDecompressAsync",
    "LZ4": "Zstd",
    "lz4": "zstd",
}
for a, b in repls.items():
    cpp = cpp.replace(a, b)

(root / "models/lbrc_nvcomp_zstd_batched.cpp").write_text(cpp)

tool = (root / "tools/caesar_lbrc_nvcomp_batched.cpp").read_text()
tool = tool.replace('#include "models/lbrc_nvcomp_batched.h"', '#include "models/lbrc_nvcomp_zstd_batched.h"')
tool = tool.replace("caesar::lbrc_nvcomp_batched::", "caesar::lbrc_nvcomp_zstd_batched::")
tool = tool.replace("batched nvCOMP LZ4", "batched nvCOMP Zstd")
(root / "tools/caesar_lbrc_nvcomp_zstd_batched.cpp").write_text(tool)
