from pathlib import Path
import csv, math
import matplotlib.pyplot as plt

OUT = Path("/blue/ranka/sa.nandikanti/residual_runs/lbrc_cpp_validation_all_eb")
SUMMARY = OUT / "compare_summary.tsv"
PNG = OUT / "lbrc_python_vs_cpp_all_datasets_all_eb_visible.png"

rows = []
with SUMMARY.open(newline="") as f:
    for r in csv.DictReader(f, delimiter="\t"):
        rows.append({
            "dataset": r["dataset"],
            "method": r["method"],
            "nrmse": float(r["final_nrmse"]),
            "cr": float(r["cr"]),
        })

order = ["E3SM","JHTDB","ERA5","Hurricane","HYCOM","Openfoam","PDEBench","S3D","TUM_TF"]
datasets = [d for d in order if any(r["dataset"] == d for r in rows)]

fig, axes = plt.subplots(math.ceil(len(datasets)/3), 3, figsize=(16, 14), squeeze=False)

for ax, ds in zip(axes.flat, datasets):
    py = sorted([r for r in rows if r["dataset"] == ds and r["method"] == "python"], key=lambda r: r["nrmse"])
    cc = sorted([r for r in rows if r["dataset"] == ds and r["method"] == "cpp"], key=lambda r: r["nrmse"])

    if cc:
        ax.plot([r["nrmse"] for r in cc], [r["cr"] for r in cc],
                linestyle="--", marker="x", linewidth=1.8, markersize=7,
                color="#DD8452", label="C++ LBRC", zorder=2)

    if py:
        ax.plot([r["nrmse"] for r in py], [r["cr"] for r in py],
                linestyle="-", marker="o", linewidth=1.4, markersize=8,
                markerfacecolor="white", markeredgewidth=2.0,
                color="#4C72B0", label="Python LBRC", zorder=3)

    ax.set_xscale("log")
    ax.set_title(ds, fontweight="bold")
    ax.set_xlabel("Real NRMSE")
    ax.set_ylabel("Compression Ratio")
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=9)

for ax in axes.flat[len(datasets):]:
    ax.axis("off")

fig.suptitle("LBRC Python vs C++: Compression Ratio vs Real NRMSE", fontsize=16, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(PNG, dpi=220)
print("Wrote", PNG)
