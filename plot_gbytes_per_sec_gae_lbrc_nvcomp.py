import csv
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BASE = Path("/blue/ranka/sa.nandikanti/residual_runs")
LBRC_SWEEP = BASE / "lbrc_cpp_validation_nvcomp_zstd_sweep/cpu_zstd_vs_nvcomp_zstd_all_eb.tsv"
OUTDIR = BASE / "lbrc_cpp_validation_nvcomp_zstd_sweep"

# Put GAE timing rows here if they are not already available from logs.
# Columns: dataset eb method seconds
GAE_TIMES = OUTDIR / "gae_times.tsv"

DATASET_NPZ = {
    "E3SM": "/blue/ranka/sa.nandikanti/residual_runs/residual_prepare/e3sm_test/e3sm_residual_input.npz",
    "JHTDB": "/blue/ranka/shared-liangji/RC_Data/jhtdb.npz",
    "ERA5": "/blue/ranka/sa.nandikanti/residual_runs/residual_prepare_all/prepared/ERA5_residual_input.npz",
    "Hurricane": "/blue/ranka/sa.nandikanti/residual_runs/residual_prepare_all/prepared/Hurricane_residual_input.npz",
    "HYCOM": "/blue/ranka/sa.nandikanti/residual_runs/residual_prepare_all/prepared/HYCOM_residual_input.npz",
    "Openfoam": "/blue/ranka/sa.nandikanti/residual_runs/residual_prepare_all/prepared/Openfoam_residual_input.npz",
    "PDEBench": "/blue/ranka/sa.nandikanti/residual_runs/residual_prepare_all/prepared/PDEBench_residual_input.npz",
    "S3D": "/blue/ranka/sa.nandikanti/residual_runs/residual_prepare_all/prepared/S3D_residual_input.npz",
    "TUM_TF": "/blue/ranka/sa.nandikanti/residual_runs/residual_prepare_all/prepared/TUM_TF_residual_input.npz",
}

def seconds_from_wall(s):
    parts = str(s).strip().split(":")
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    return float(s)

def original_gbytes(dataset):
    import numpy as np
    z = np.load(DATASET_NPZ[dataset])
    arr = z["original_data"]
    return arr.size * np.dtype("float32").itemsize / 1e9

rows = []

with open(LBRC_SWEEP, newline="") as f:
    for r in csv.DictReader(f, delimiter="\t"):
        ds = r["dataset"]
        eb = r["eb"]
        method = r["method"]
        gb = original_gbytes(ds)

        # Use measured LBRC internal time for compressor throughput.
        sec = float(r["lbrc_time_s"])
        rows.append({
            "dataset": ds,
            "eb": eb,
            "method": method,
            "gbps": gb / sec if sec > 0 else math.nan,
        })

if not GAE_TIMES.exists():
    GAE_TIMES.write_text("dataset\teb\tmethod\tseconds\n")
    print(f"Wrote template: {GAE_TIMES}")
    print("Add GAE timing rows there, then rerun this script.")
else:
    with open(GAE_TIMES, newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if not r.get("dataset") or r["dataset"].startswith("#"):
                continue
            ds = r["dataset"]
            gb = original_gbytes(ds)
            sec = float(r["seconds"])
            rows.append({
                "dataset": ds,
                "eb": r["eb"],
                "method": r.get("method", "GAE") or "GAE",
                "gbps": gb / sec if sec > 0 else math.nan,
            })

summary = OUTDIR / "gbytes_per_sec_gae_lbrc_nvcomp.tsv"
with open(summary, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["dataset", "eb", "method", "gbps"], delimiter="\t")
    w.writeheader()
    for r in rows:
        w.writerow(r)

datasets = list(DATASET_NPZ)
eb_order = ["1e-3", "3e-4", "1e-4", "3e-5", "1e-5", "3e-6", "1e-6"]
methods = ["GAE", "cpu_zstd", "nvcomp_zstd"]
labels = {"GAE": "GAE", "cpu_zstd": "LBRC CPU Zstd", "nvcomp_zstd": "LBRC nvCOMP Zstd"}
colors = {"GAE": "#4C78A8", "cpu_zstd": "#F58518", "nvcomp_zstd": "#54A24B"}

fig, axes = plt.subplots(3, 3, figsize=(18, 14), constrained_layout=True)
axes = axes.ravel()

for ax, ds in zip(axes, datasets):
    subset = [r for r in rows if r["dataset"] == ds]
    for m in methods:
        pts = []
        for eb in eb_order:
            vals = [r["gbps"] for r in subset if r["method"] == m and r["eb"] == eb]
            pts.append(vals[0] if vals else np.nan)
        if np.isfinite(pts).any():
            ax.plot(eb_order, pts, marker="o", linewidth=2, label=labels[m], color=colors[m])
    ax.set_title(ds, fontweight="bold")
    ax.set_xlabel("Error Bound / Target NRMSE")
    ax.set_ylabel("Throughput (GB/s)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="x", rotation=35)

for ax in axes[len(datasets):]:
    ax.axis("off")

handles, legend_labels = axes[0].get_legend_handles_labels()
fig.legend(handles, legend_labels, loc="upper center", ncol=3)
fig.suptitle("GAE vs LBRC CPU Zstd vs LBRC nvCOMP Zstd: Compression Throughput", fontsize=18, fontweight="bold")

out = OUTDIR / "gbytes_per_sec_gae_lbrc_nvcomp.png"
fig.savefig(out, dpi=220)
print("Wrote", summary)
print("Wrote", out)
