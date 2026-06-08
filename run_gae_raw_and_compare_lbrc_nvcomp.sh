#!/usr/bin/env bash
set -euo pipefail
unset LD_PRELOAD

export CAESAR_HOME=/blue/ranka/sa.nandikanti/CAESAR/install-jupyter
export GCC_LIB=/apps/compilers/gcc/14.2.0/lib64
export TORCH_LIB=/apps/jupyter/6.5.4/lib/python3.10/site-packages/torch/lib
export CUDA_HOME=/apps/compilers/cuda/12.8.1
export NVCOMP_ROOT=/home/sa.nandikanti/local/nvcomp

export LD_LIBRARY_PATH=$GCC_LIB:$CAESAR_HOME/lib:$NVCOMP_ROOT/lib:$TORCH_LIB:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export PATH=$CAESAR_HOME/bin:$CUDA_HOME/bin:/usr/bin:/bin:$PATH

PY=/apps/jupyter/6.5.4/bin/python3
OUT=/blue/ranka/sa.nandikanti/residual_runs/gae_raw_vs_lbrc_nvcomp_zstd
RAW=$OUT/raw
mkdir -p "$RAW" "$OUT/logs"

cat > "$OUT/datasets.tsv" <<'TSV'
dataset	npz	shape
E3SM	/blue/ranka/sa.nandikanti/residual_runs/residual_prepare/e3sm_test/e3sm_residual_input.npz	1,6,720,240,240
JHTDB	/blue/ranka/shared-liangji/RC_Data/jhtdb.npz	1,4,240,512,512
ERA5	/blue/ranka/sa.nandikanti/residual_runs/residual_prepare_all/prepared/ERA5_residual_input.npz	1,1,960,512,512
Hurricane	/blue/ranka/sa.nandikanti/residual_runs/residual_prepare_all/prepared/Hurricane_residual_input.npz	1,16,47,500,500
HYCOM	/blue/ranka/sa.nandikanti/residual_runs/residual_prepare_all/prepared/HYCOM_residual_input.npz	1,64,128,256,256
Openfoam	/blue/ranka/sa.nandikanti/residual_runs/residual_prepare_all/prepared/Openfoam_residual_input.npz	1,4,240,512,512
PDEBench	/blue/ranka/sa.nandikanti/residual_runs/residual_prepare_all/prepared/PDEBench_residual_input.npz	1,8,128,512,512
S3D	/blue/ranka/sa.nandikanti/residual_runs/residual_prepare_all/prepared/S3D_residual_input.npz	8,1,50,640,640
TUM_TF	/blue/ranka/sa.nandikanti/residual_runs/residual_prepare_all/prepared/TUM_TF_residual_input.npz	1,16,240,256,256
TSV

EBS=("1e-3" "3e-4" "1e-4" "3e-5" "1e-5" "3e-6" "1e-6")

echo -e "dataset\teb\tseconds\tgbps\tcr\twall_s\tmaxrss_gib" > "$OUT/gae_raw_bench.tsv"

tail -n +2 "$OUT/datasets.tsv" | while IFS=$'\t' read -r DATASET NPZ SHAPE; do
  ORIG="$RAW/${DATASET}_original.bin"
  GB_FILE="$RAW/${DATASET}_gb.txt"

  echo "===== prepare raw $DATASET ====="
  "$PY" - <<PY
import numpy as np
from pathlib import Path
z = np.load("$NPZ")
a = z["original_data"].astype("float32", copy=False)
Path("$GB_FILE").write_text(str(a.size * 4 / 1e9))
p = Path("$ORIG")
if not p.exists():
    a.tofile(p)
print("$DATASET", a.shape, "GB", a.size * 4 / 1e9)
PY

  GB=$(cat "$GB_FILE")

  for EB in "${EBS[@]}"; do
    LOG="$OUT/logs/${DATASET}_gae_${EB}.log"
    echo "===== GAE raw $DATASET EB=$EB ====="

    /usr/bin/time -v caesar_gae_bench \
      --input "$ORIG" \
      --shape "$SHAPE" \
      --nrmse "$EB" \
      --batch-size 32 \
      --n-frame 8 \
      > "$LOG" 2>&1

    "$PY" - <<PY
import re
from pathlib import Path

log = Path("$LOG").read_text(errors="ignore")

def grab(pattern, default="nan"):
    m = re.search(pattern, log)
    return m.group(1) if m else default

sec = float(grab(r"GAE time:\\s*([0-9.eE+-]+)"))
cr = grab(r"GAE CR latent_only:\\s*([0-9.eE+-]+)")
wall = grab(r"Elapsed \\(wall clock\\) time.*?:\\s*([^\\n]+)", "")
rss_kb = float(grab(r"Maximum resident set size \\(kbytes\\):\\s*([0-9]+)", "0"))
gbps = float("$GB") / sec
rss_gib = rss_kb / 1024 / 1024

with open("$OUT/gae_raw_bench.tsv", "a") as f:
    f.write(f"$DATASET\\t$EB\\t{sec}\\t{gbps}\\t{cr}\\t{wall}\\t{rss_gib:.3f}\\n")

print("$DATASET", "$EB", "seconds", sec, "GB/s", gbps, "CR", cr)
PY
  done
done

echo "Wrote $OUT/gae_raw_bench.tsv"
