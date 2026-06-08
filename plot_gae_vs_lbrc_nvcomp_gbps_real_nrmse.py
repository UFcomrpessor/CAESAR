import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BASE = Path("/blue/ranka/sa.nandikanti/residual_runs")
PREP = BASE / "residual_prepare_all"
OUTDIR = BASE / "lbrc_cpp_validation_nvcomp_zstd_sweep"

GAE_NRMSE = PREP / "caesar_real_nrmse_summary.tsv"
GAE_TIMES = OUTDIR / "gae_times.tsv"
NVCOMP_SWEEP = OUTDIR / "cpu_zstd_vs_nvcomp_zstd_all_eb.tsv"

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

def original_gb(dataset):
    import numpy as np
    z = np.load(DATASET_NPZ[dataset])
    a = z["original_data"]
    return a.size * np.dtype("float32").itemsize / 1e9

def read_tsv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))

gae_real = {}
for r in read_tsv(GAE_NRMSE):
    gae_real[(r["dataset"], r["eb"])] = float(r["real_nrmse"])

gae_time = {}
for r in read_tsv(GAE_TIMES):
    if not r.get("seconds") or "PUT" in r["seconds"]:
        continue
    gae_time[(r["dataset"], r["eb"])] = float(r["seconds"])

rows = []

for (dataset, eb), sec in gae_time.items():
    if (dataset, eb) not in gae_real:
        continue
    rows.append({
        "dataset": dataset,
        "eb": eb,
        "method": "GAE",
        "real_nrmse": gae_real[(dataset, eb)],
        "gbps": original_gb(dataset) / sec,
    })

for r in read_tsv(NVCOMP_SWEEP):
    if r["method"] != "nvcomp_zstd":
        continue
    dataset = r["dataset"]
    sec = float(r["lbrc_time_s"])
    rows.append({
        "dataset": dataset,
        "eb": r["eb"],
        "method": "LBRC nvCOMP Zstd",
        "real_nrmse": float(r["final_nrmse"]),
        "gbps": original_gb(dataset) / sec,
    })

if not any(r["method"] == "GAE" for r in rows):
    raise SystemExit(f"No GAE points. Make sure {GAE_TIMES} has measured seconds.")
if not any(r["method"] == "LBRC nvCOMP Zstd" for r in rows):
    raise SystemExit(f"No LBRC nvCOMP Zstd points found in {NVCOMP_SWEEP}.")

summary = OUTDIR / "gae_vs_lbrc_nvcomp_gbps_real_nrmse.tsv"
with open(summary, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["dataset", "eb", "method", "real_nrmse", "gbps"], delimiter="\t")
    w.writeheader()
    w.writerows(rows)

fig, axes = plt.subplots(3, 3, figsize=(18, 14), constrained_layout=True)
axes = axes.ravel()

styles = {
    "GAE": dict(color="#4C78A8", marker="o", linewidth=2.2),
    "LBRC nvCOMP Zstd": dict(color="#54A24B", marker="s", linewidth=2.2),
}

for ax, dataset in zip(axes, DATASET_NPZ):
    subset = [r for r in rows if r["dataset"] == dataset]

    for method, style in styles.items():
        pts = [r for r in subset if r["method"] == method and r["real_nrmse"] > 0 and r["gbps"] > 0]
        pts.sort(key=lambda x: x["real_nrmse"])
        if pts:
            ax.plot(
                [p["real_nrmse"] for p in pts],
                [p["gbps"] for p in pts],
                label=method,
                **style,
            )

    ax.set_title(dataset, fontweight="bold")
    ax.set_xlabel("Real NRMSE")
    ax.set_ylabel("Throughput (GB/s)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

for ax in axes[len(DATASET_NPZ):]:
    ax.axis("off")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=2)
fig.suptitle("GAE vs LBRC nvCOMP Zstd: Throughput vs Real NRMSE", fontsize=18, fontweight="bold")

out = OUTDIR / "gae_vs_lbrc_nvcomp_gbps_real_nrmse.png"
fig.savefig(out, dpi=220)

print("Wrote", summary)
print("Wrote", out)
