#!/usr/bin/env bash
set -euo pipefail

PY=/apps/jupyter/6.5.4/bin/python3
GCC_HOME=/apps/compilers/gcc/14.2.0
GXX=$GCC_HOME/bin/g++
export LD_LIBRARY_PATH=$GCC_HOME/lib64:${LD_LIBRARY_PATH:-}
CAESAR_LBRC=/blue/ranka/sa.nandikanti/CAESAR/install-jupyter/bin/caesar_lbrc
LBRC_DIR=/blue/ranka/sa.nandikanti/residual_runs/residual-modeling/LBRC
VALID=/blue/ranka/sa.nandikanti/residual_runs/lbrc_cpp_validation
SRC=/blue/ranka/sa.nandikanti/residual_runs/residual_prepare_all/prepared/ERA5_residual_input.npz

mkdir -p "$VALID/logs"

echo "== Build ASan standalone binary =="
"$GXX" -std=c++17 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer \
  -I /blue/ranka/sa.nandikanti/CAESAR/CAESAR \
  /blue/ranka/sa.nandikanti/CAESAR/CAESAR/models/lbrc.cpp \
  /blue/ranka/sa.nandikanti/CAESAR/CAESAR/tools/caesar_lbrc.cpp \
  -lzstd -pthread -o "$VALID/caesar_lbrc_asan"

echo "== Prepare small ERA5 slice =="
"$PY" - <<PY
import numpy as np
from pathlib import Path

src = "$SRC"
out = Path("$VALID")
z = np.load(src)
orig = z["original_data"][:, :, :60, :128, :128].astype("float32", copy=True)
rec = z["recons_data"][:, :, :60, :128, :128].astype("float32", copy=True)
np.savez(out / "era5_small.npz", original_data=orig, recons_data=rec, latent_bit=np.array(0, dtype=np.int64))
orig.tofile(out / "era5_small_original.bin")
rec.tofile(out / "era5_small_recons.bin")
(out / "era5_small_shape.txt").write_text(",".join(map(str, orig.shape)))
print("shape", orig.shape, "elements", orig.size)
PY

SHAPE=$(cat "$VALID/era5_small_shape.txt")

echo "== Python LBRC small =="
cd "$LBRC_DIR"
/usr/bin/time -v "$PY" -u LBRC.py \
  --mode evl \
  --path "$VALID/era5_small.npz" \
  --nrmse 1e-5 \
  --block_t 60 \
  --block_h 128 \
  --block_w 128 \
  --level 21 \
  --quant_iter 16 \
  --workers 16 \
  > "$VALID/logs/python_era5_small.log" 2>&1

echo "== C++ LBRC small =="
/usr/bin/time -v "$CAESAR_LBRC" \
  --original "$VALID/era5_small_original.bin" \
  --recons "$VALID/era5_small_recons.bin" \
  --shape "$SHAPE" \
  --latent-bit 0 \
  --nrmse 1e-5 \
  --block-t 60 \
  --block-h 128 \
  --block-w 128 \
  --level 21 \
  --quant-iter 16 \
  --workers 16 \
  > "$VALID/logs/cpp_era5_small.log" 2>&1

echo "== ASan/UBSan C++ small =="
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
/usr/bin/time -v "$VALID/caesar_lbrc_asan" \
  --original "$VALID/era5_small_original.bin" \
  --recons "$VALID/era5_small_recons.bin" \
  --shape "$SHAPE" \
  --latent-bit 0 \
  --nrmse 1e-5 \
  --block-t 60 \
  --block-h 128 \
  --block-w 128 \
  --level 21 \
  --quant-iter 16 \
  --workers 4 \
  > "$VALID/logs/asan_era5_small.log" 2>&1

echo "== Key lines =="
for f in python_era5_small cpp_era5_small asan_era5_small; do
  echo "--- $f ---"
  /usr/bin/grep -n "Target NRMSE\|Final NRMSE\|Encoded NRMSE\|CR:\|LBRC time\|Elapsed (wall clock)\|Maximum resident\|ERROR: AddressSanitizer\|runtime error\|Traceback\|Error" \
    "$VALID/logs/${f}.log" || true
done
