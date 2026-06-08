import csv
from pathlib import Path

import matplotlib.pyplot as plt

BASE = Path("/blue/ranka/sa.nandikanti/residual_runs")
OUT = BASE / "gae_raw_vs_lbrc_nvcomp_zstd"
GAE = BASE / "residual_prepare_all/caesar_real_nrmse_summary.tsv"
LBRC = BASE / "lbrc_cpp_validation_nvcomp_zstd_sweep/cpu_zstd_vs_nvcomp_zstd_all_eb.tsv"

DATASETS = ["E3SM", "JHTDB", "ERA5", "Hurricane", "HYCOM", "Openfoam", "PDEBench", "S3D", "TUM_TF"]
LO, HI = 9e-7, 1e-3

def read_tsv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))

rows = []

for r in read_tsv(GAE):
    rn = float(r["real_nrmse"])
    if LO <= rn <= HI:
        rows.append({
            "dataset": r["dataset"],
            "eb": r["eb"],
            "method": "GAE",
            "real_nrmse": rn,
            "cr": float(r["CR"]),
        })

for r in read_tsv(LBRC):
    if r["method"] != "nvcomp_zstd":
        continue
    rn = float(r["final_nrmse"])
    if LO <= rn <= HI:
        rows.append({
            "dataset": r["dataset"],
            "eb": r["eb"],
            "method": "LBRC nvCOMP Zstd",
            "real_nrmse": rn,
            "cr": float(r["cr"]),
        })

summary = OUT / "gae_summary_cr_vs_lbrc_nvcomp.tsv"
with open(summary, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["dataset", "eb", "method", "real_nrmse", "cr"], delimiter="\t")
    w.writeheader()
    w.writerows(rows)

fig, axes = plt.subplots(3, 3, figsize=(18, 14), constrained_layout=True)
axes = axes.ravel()

styles = {
    "GAE": dict(color="#4C78A8", marker="o", linewidth=2.2),
    "LBRC nvCOMP Zstd": dict(color="#54A24B", marker="s", linewidth=2.2),
}

for ax, ds in zip(axes, DATASETS):
    subset = [r for r in rows if r["dataset"] == ds]
    for method, style in styles.items():
        pts = [r for r in subset if r["method"] == method]
        pts.sort(key=lambda x: x["real_nrmse"])
        if pts:
            ax.plot([p["real_nrmse"] for p in pts], [p["cr"] for p in pts], label=method, **style)

    ax.set_title(ds, fontweight="bold")
    ax.set_xlabel("Real NRMSE")
    ax.set_ylabel("Compression Ratio")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(LO, HI)
    ax.grid(True, alpha=0.3)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=2)
fig.suptitle("GAE vs LBRC nvCOMP Zstd: Compression Ratio vs Real NRMSE", fontsize=18, fontweight="bold")

out = OUT / "gae_summary_cr_vs_lbrc_nvcomp.png"
fig.savefig(out, dpi=220)

print("Wrote", summary)
print("Wrote", out)
