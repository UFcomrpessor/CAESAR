#!/usr/bin/env bash
set -euo pipefail
unset LD_PRELOAD

PY=/apps/jupyter/6.5.4/bin/python3
CAESAR_LBRC=/blue/ranka/sa.nandikanti/CAESAR/install-jupyter/bin/caesar_lbrc
LBRC_DIR=/blue/ranka/sa.nandikanti/residual_runs/residual-modeling/LBRC
OUT=/blue/ranka/sa.nandikanti/residual_runs/lbrc_cpp_validation_all_eb
mkdir -p "$OUT/raw" "$OUT/logs"

cat > "$OUT/datasets.tsv" <<'TSV'
dataset	npz	shape	bt	bh	bw
E3SM	/blue/ranka/sa.nandikanti/residual_runs/residual_prepare/e3sm_test/e3sm_residual_input.npz	1,6,240,240,240	60	120	120
JHTDB	/blue/ranka/shared-liangji/RC_Data/jhtdb.npz	1,4,240,512,512	60	128	128
ERA5	/blue/ranka/sa.nandikanti/residual_runs/residual_prepare_all/prepared/ERA5_residual_input.npz	1,1,960,512,512	60	128	128
Hurricane	/blue/ranka/sa.nandikanti/residual_runs/residual_prepare_all/prepared/Hurricane_residual_input.npz	1,16,47,500,500	47	125	125
HYCOM	/blue/ranka/sa.nandikanti/residual_runs/residual_prepare_all/prepared/HYCOM_residual_input.npz	1,64,128,256,256	64	128	128
Openfoam	/blue/ranka/sa.nandikanti/residual_runs/residual_prepare_all/prepared/Openfoam_residual_input.npz	1,4,240,512,512	60	128	128
PDEBench	/blue/ranka/sa.nandikanti/residual_runs/residual_prepare_all/prepared/PDEBench_residual_input.npz	1,8,128,512,512	64	128	128
S3D	/blue/ranka/sa.nandikanti/residual_runs/residual_prepare_all/prepared/S3D_residual_input.npz	8,1,50,640,640	50	128	128
TUM_TF	/blue/ranka/sa.nandikanti/residual_runs/residual_prepare_all/prepared/TUM_TF_residual_input.npz	1,16,240,256,256	60	128	128
TSV

EBS=("1e-3" "3e-4" "1e-4" "3e-5" "1e-5" "3e-6" "1e-6")
echo -e "dataset\teb\tmethod\tfinal_nrmse\tcr\twall_s\tmaxrss_gib" > "$OUT/compare_summary.tsv"

tail -n +2 "$OUT/datasets.tsv" | while IFS=$'\t' read -r DATASET NPZ SHAPE BT BH BW; do
  echo "===== prepare $DATASET ====="
  ORIG="$OUT/raw/${DATASET}_original.bin"
  RECONS="$OUT/raw/${DATASET}_recons.bin"
  LATENT="$OUT/raw/${DATASET}_latent_bit.txt"
  NPZ_COPY="$OUT/raw/${DATASET}_residual_input.npz"

  "$PY" - <<PY
import numpy as np
from pathlib import Path
z = np.load("$NPZ")
orig = z["original_data"].astype("float32", copy=False)
rec = z["recons_data"].astype("float32", copy=False)
latent = int(np.asarray(z["latent_bit"]).item()) if "latent_bit" in z.files else 0
if not Path("$ORIG").exists(): orig.tofile("$ORIG")
if not Path("$RECONS").exists(): rec.tofile("$RECONS")
Path("$LATENT").write_text(str(latent))
if not Path("$NPZ_COPY").exists():
    np.savez("$NPZ_COPY", original_data=orig, recons_data=rec, latent_bit=np.array(latent, dtype=np.int64))
print(orig.shape, "latent_bit", latent)
PY
  LATENT_BIT=$(cat "$LATENT")

  for EB in "${EBS[@]}"; do
    echo "===== $DATASET EB=$EB ====="

    cd "$LBRC_DIR"
    /usr/bin/time -v "$PY" -u LBRC.py \
      --mode evl --path "$NPZ_COPY" --nrmse "$EB" \
      --block_t "$BT" --block_h "$BH" --block_w "$BW" \
      --level 21 --quant_iter 16 --workers 16 \
      > "$OUT/logs/${DATASET}_python_${EB}.log" 2>&1

    /usr/bin/time -v "$CAESAR_LBRC" \
      --original "$ORIG" --recons "$RECONS" --shape "$SHAPE" \
      --latent-bit "$LATENT_BIT" --nrmse "$EB" \
      --block-t "$BT" --block-h "$BH" --block-w "$BW" \
      --level 21 --quant-iter 16 --workers 16 \
      > "$OUT/logs/${DATASET}_cpp_${EB}.log" 2>&1

    "$PY" - <<PY >> "$OUT/compare_summary.tsv"
from pathlib import Path
import re
def parse(path):
    text = Path(path).read_text(errors="ignore")
    n = re.findall(r"Final NRMSE:\s*([0-9.eE+-]+)", text)
    c = re.findall(r"CR:\s*([0-9.eE+-]+)", text)
    w = re.findall(r"Elapsed \\(wall clock\\) time.*?:\\s*([0-9:.]+)", text)
    m = re.findall(r"Maximum resident set size \\(kbytes\\):\\s*([0-9]+)", text)
    return n[-1], c[-1], (w[-1] if w else "NA"), (float(m[-1])/1024**2 if m else float("nan"))
for method, log in [("python", "$OUT/logs/${DATASET}_python_${EB}.log"), ("cpp", "$OUT/logs/${DATASET}_cpp_${EB}.log")]:
    n,c,w,r = parse(log)
    print(f"$DATASET\t$EB\t{method}\t{n}\t{c}\t{w}\t{r:.3f}")
PY
  done
done

column -t -s $'\t' "$OUT/compare_summary.tsv"
