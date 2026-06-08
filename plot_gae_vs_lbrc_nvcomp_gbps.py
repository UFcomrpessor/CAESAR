import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BASE = Path("/blue/ranka/sa.nandikanti/residual_runs")
SWEEP = BASE / "lbrc_cpp_validation_nvcomp_zstd_sweep/cpu_zstd_vs_nvcomp_zstd_all_eb.tsv"
OUTDIR = BASE / "lbrc_cpp_validation_nvcomp_zstd_sweep"
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

EB_ORDER = ["1e-3", "3e-4", "1e-4", "3e-5", "1e-5", "3e-6", "1e-6"]

def original_gb(dataset):
    import numpy as np
    z = np.load(DATASET_NPZ[dataset])
    a = z["original_data"]
    return a.size * np.dtype("float32").itemsize / 1e9

def make_template():
    with open(GAE_TIMES, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["dataset", "eb", "seconds"])
        for ds in DATASET_NPZ:
            for eb in EB_ORDER:
                w.writerow([ds, eb, "PUT_GAE_SECONDS_HERE"])

rows = []

if not GAE_TIMES.exists():
    make_template()
    raise SystemExit(f"Created template {GAE_TIMES}. Fill GAE seconds, then rerun.")

with open(GAE_TIMES, newline="") as f:
    for r in csv.DictReader(f, delimiter="\t"):
        if not r["seconds"] or "PUT" in r["seconds"]:
            continue
        ds = r["dataset"]
        sec = float(r["seconds"])
        rows.append({
            "dataset": ds,
            "eb": r["eb"],
            "method": "GAE",
            "gbps": original_gb(ds) / sec,
        })

with open(SWEEP, newline="") as f:
    for r in csv.DictReader(f, delimiter="\t"):
        if r["method"] != "nvcomp_zstd":
            continue
        ds = r["dataset"]
        sec = float(r["lbrc_time_s"])
        rows.append({
            "dataset": ds,
            "eb": r["eb"],
            "method": "LBRC nvCOMP Zstd",
            "gbps": original_gb(ds) / sec,
        })

gae_count = sum(1 for r in rows if r["method"] == "GAE")
nv_count = sum(1 for r in rows if r["method"] == "LBRC nvCOMP Zstd")
if gae_count == 0:
    raise SystemExit(f"No GAE rows found in {GAE_TIMES}. Fill the seconds column first.")
if nv_count == 0:
    raise SystemExit(f"No nvcomp_zstd rows found in {SWEEP}.")

summary = OUTDIR / "gae_vs_lbrc_nvcomp_gbytes_per_sec.tsv"
with open(summary, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["dataset", "eb", "method", "gbps"], delimiter="\t")
    w.writeheader()
    w.writerows(rows)

fig, axes = plt.subplots(3, 3, figsize=(18, 14), constrained_layout=True)
axes = axes.ravel()

styles = {
    "GAE": dict(color="#4C78A8", marker="o", linewidth=2.2),
    "LBRC nvCOMP Zstd": dict(color="#54A24B", marker="s", linewidth=2.2),
}

for ax, ds in zip(axes, DATASET_NPZ):
    subset = [r for r in rows if r["dataset"] == ds]
    for method, style in styles.items():
        y = []
        for eb in EB_ORDER:
            vals = [r["gbps"] for r in subset if r["method"] == method and r["eb"] == eb]
            y.append(vals[0] if vals else np.nan)
        if np.isfinite(y).any():
            ax.plot(EB_ORDER, y, label=method, **style)

    ax.set_title(ds, fontweight="bold")
    ax.set_xlabel("Error Bound / Target NRMSE")
    ax.set_ylabel("Throughput (GB/s)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="x", rotation=35)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=2)
fig.suptitle("GAE vs LBRC nvCOMP Zstd: Compression Throughput", fontsize=18, fontweight="bold")

out = OUTDIR / "gae_vs_lbrc_nvcomp_gbytes_per_sec.png"
fig.savefig(out, dpi=220)
print("GAE rows:", gae_count)
print("LBRC nvCOMP rows:", nv_count)
print("Wrote", summary)
print("Wrote", out)
