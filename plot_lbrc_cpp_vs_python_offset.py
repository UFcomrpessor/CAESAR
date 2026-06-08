from pathlib import Path
import csv
import math
import matplotlib.pyplot as plt

OUT = Path("/blue/ranka/sa.nandikanti/residual_runs/lbrc_cpp_validation_all_eb")
SUMMARY = OUT / "compare_summary.tsv"
PNG = OUT / "lbrc_python_vs_cpp_all_datasets_all_eb_offset.png"

rows = []
with SUMMARY.open(newline="") as f:
    for r in csv.DictReader(f, delimiter="\t"):
        if r["final_nrmse"] in ("", "NA") or r["cr"] in ("", "NA"):
            continue
        rows.append((r["dataset"], r["method"], float(r["final_nrmse"]), float(r["cr"])))

order = ["E3SM","JHTDB","ERA5","Hurricane","HYCOM","Openfoam","PDEBench","S3D","TUM_TF"]
fig, axes = plt.subplots(math.ceil(len(order) / 3), 3, figsize=(16, 14), squeeze=False)

for ax, ds in zip(axes.flat, order):
    for method, color, marker, shift in [
        ("python", "#4C72B0", "o", 0.97),
        ("cpp", "#DD8452", "x", 1.03),
    ]:
        sub = sorted([(x, y) for d, m, x, y in rows if d == ds and m == method])
        if not sub:
            continue
        ax.plot(
            [x * shift for x, y in sub],
            [y for x, y in sub],
            marker=marker,
            linewidth=1.8,
            markersize=7,
            label=("Python LBRC" if method == "python" else "C++ LBRC"),
            color=color,
        )

    ax.set_xscale("log")
    ax.set_title(ds, fontweight="bold")
    ax.set_xlabel("Real NRMSE")
    ax.set_ylabel("Compression Ratio")
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=9)

for ax in axes.flat[len(order):]:
    ax.axis("off")

fig.suptitle("LBRC Python vs C++ (x shifted +/-3% only to reveal overlap)", fontsize=16, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(PNG, dpi=220)
print("Wrote", PNG)
