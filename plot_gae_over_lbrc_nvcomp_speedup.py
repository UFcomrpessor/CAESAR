import csv
from pathlib import Path

import matplotlib.pyplot as plt

OUTDIR = Path("/blue/ranka/sa.nandikanti/residual_runs/lbrc_cpp_validation_nvcomp_zstd_sweep")
IN = OUTDIR / "gae_vs_lbrc_nvcomp_gbps_real_nrmse_filtered.tsv"

DATASETS = ["E3SM", "JHTDB", "ERA5", "Hurricane", "HYCOM", "Openfoam", "PDEBench", "S3D", "TUM_TF"]

rows = []
with open(IN, newline="") as f:
    for r in csv.DictReader(f, delimiter="\t"):
        rows.append(r)

by = {}
for r in rows:
    key = (r["dataset"], r["eb"])
    by.setdefault(key, {})[r["method"]] = r

speedups = []
for (dataset, eb), methods in by.items():
    if "GAE" not in methods or "LBRC nvCOMP Zstd" not in methods:
        continue

    gae = methods["GAE"]
    lbrc = methods["LBRC nvCOMP Zstd"]

    gae_gbps = float(gae["gbps"])
    lbrc_gbps = float(lbrc["gbps"])

    if gae_gbps <= 0 or lbrc_gbps <= 0:
        continue

    # Use the GAE real NRMSE as the x-coordinate for the paired comparison.
    speedups.append({
        "dataset": dataset,
        "eb": eb,
        "real_nrmse": float(gae["real_nrmse"]),
        "speedup": gae_gbps / lbrc_gbps,
        "gae_gbps": gae_gbps,
        "lbrc_gbps": lbrc_gbps,
    })

summary = OUTDIR / "gae_over_lbrc_nvcomp_speedup.tsv"
with open(summary, "w", newline="") as f:
    w = csv.DictWriter(
        f,
        fieldnames=["dataset", "eb", "real_nrmse", "speedup", "gae_gbps", "lbrc_gbps"],
        delimiter="\t",
    )
    w.writeheader()
    w.writerows(speedups)

fig, axes = plt.subplots(3, 3, figsize=(18, 14), constrained_layout=True)
axes = axes.ravel()

for ax, ds in zip(axes, DATASETS):
    pts = [r for r in speedups if r["dataset"] == ds]
    pts.sort(key=lambda x: x["real_nrmse"])

    if pts:
        ax.plot(
            [p["real_nrmse"] for p in pts],
            [p["speedup"] for p in pts],
            marker="o",
            linewidth=2.2,
            color="#7F3C8D",
        )

    ax.axhline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_title(ds, fontweight="bold")
    ax.set_xlabel("GAE Real NRMSE")
    ax.set_ylabel("Speedup: GAE / LBRC nvCOMP Zstd")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

fig.suptitle("GAE Throughput Speedup over LBRC nvCOMP Zstd", fontsize=18, fontweight="bold")

out = OUTDIR / "gae_over_lbrc_nvcomp_speedup.png"
fig.savefig(out, dpi=220)

print("Wrote", summary)
print("Wrote", out)
