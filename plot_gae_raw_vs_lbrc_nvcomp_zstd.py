import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BASE = Path("/blue/ranka/sa.nandikanti/residual_runs")
OUT = BASE / "gae_raw_vs_lbrc_nvcomp_zstd"
PREP = BASE / "residual_prepare_all"
LBRC = BASE / "lbrc_cpp_validation_nvcomp_zstd_sweep/cpu_zstd_vs_nvcomp_zstd_all_eb.tsv"

GAE_BENCH = OUT / "gae_raw_bench.tsv"
GAE_REAL = PREP / "caesar_real_nrmse_summary.tsv"

DATASETS = ["E3SM", "JHTDB", "ERA5", "Hurricane", "HYCOM", "Openfoam", "PDEBench", "S3D", "TUM_TF"]
LO, HI = 9e-7, 1e-3

def read_tsv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))

gae_real = {(r["dataset"], r["eb"]): float(r["real_nrmse"]) for r in read_tsv(GAE_REAL)}

rows = []

for r in read_tsv(GAE_BENCH):
    key = (r["dataset"], r["eb"])
    if key not in gae_real:
        continue
    rn = gae_real[key]
    if LO <= rn <= HI:
        rows.append({
            "dataset": r["dataset"],
            "eb": r["eb"],
            "method": "GAE",
            "real_nrmse": rn,
            "gbps": float(r["gbps"]),
            "cr": float(r["cr"]),
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
            "gbps": None,  # filled from GB/s summary if available below
            "cr": float(r["cr"]),
            "lbrc_time_s": float(r["lbrc_time_s"]),
        })

# Fill LBRC GB/s using the GAE raw preparation GB files.
for r in rows:
    if r["method"] == "LBRC nvCOMP Zstd":
        gb_path = OUT / "raw" / f'{r["dataset"]}_gb.txt'
        gb = float(gb_path.read_text())
        r["gbps"] = gb / r["lbrc_time_s"]

summary = OUT / "gae_raw_vs_lbrc_nvcomp_zstd_summary.tsv"
with open(summary, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["dataset", "eb", "method", "real_nrmse", "gbps", "cr"], delimiter="\t")
    w.writeheader()
    for r in rows:
        w.writerow({k: r[k] for k in ["dataset", "eb", "method", "real_nrmse", "gbps", "cr"]})

def panel_plot(metric, ylabel, title, outfile):
    fig, axes = plt.subplots(3, 3, figsize=(18, 14), constrained_layout=True)
    axes = axes.ravel()

    styles = {
        "GAE": dict(color="#4C78A8", marker="o", linewidth=2.2),
        "LBRC nvCOMP Zstd": dict(color="#54A24B", marker="s", linewidth=2.2),
    }

    for ax, ds in zip(axes, DATASETS):
        subset = [r for r in rows if r["dataset"] == ds]
        for method, style in styles.items():
            pts = [r for r in subset if r["method"] == method and r[metric] and r[metric] > 0]
            pts.sort(key=lambda x: x["real_nrmse"])
            if pts:
                ax.plot([p["real_nrmse"] for p in pts], [p[metric] for p in pts], label=method, **style)

        ax.set_title(ds, fontweight="bold")
        ax.set_xlabel("Real NRMSE")
        ax.set_ylabel(ylabel)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(LO, HI)
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.suptitle(title, fontsize=18, fontweight="bold")
    fig.savefig(OUT / outfile, dpi=220)

panel_plot("gbps", "Throughput (GB/s)", "GAE Raw vs LBRC nvCOMP Zstd: Throughput vs Real NRMSE", "gae_raw_vs_lbrc_nvcomp_zstd_gbps.png")
panel_plot("cr", "Compression Ratio", "GAE Raw vs LBRC nvCOMP Zstd: CR vs Real NRMSE", "gae_raw_vs_lbrc_nvcomp_zstd_cr.png")

speed = []
by = {}
for r in rows:
    by.setdefault((r["dataset"], r["eb"]), {})[r["method"]] = r

for key, pair in by.items():
    if "GAE" in pair and "LBRC nvCOMP Zstd" in pair:
        speed.append({
            "dataset": key[0],
            "eb": key[1],
            "real_nrmse": pair["GAE"]["real_nrmse"],
            "speedup": pair["GAE"]["gbps"] / pair["LBRC nvCOMP Zstd"]["gbps"],
        })

speed_path = OUT / "gae_raw_over_lbrc_nvcomp_zstd_speedup.tsv"
with open(speed_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["dataset", "eb", "real_nrmse", "speedup"], delimiter="\t")
    w.writeheader()
    w.writerows(speed)

fig, axes = plt.subplots(3, 3, figsize=(18, 14), constrained_layout=True)
axes = axes.ravel()
for ax, ds in zip(axes, DATASETS):
    pts = [r for r in speed if r["dataset"] == ds]
    pts.sort(key=lambda x: x["real_nrmse"])
    if pts:
        ax.plot([p["real_nrmse"] for p in pts], [p["speedup"] for p in pts], marker="o", linewidth=2.2, color="#7F3C8D")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_title(ds, fontweight="bold")
    ax.set_xlabel("GAE Real NRMSE")
    ax.set_ylabel("Speedup: GAE / LBRC nvCOMP Zstd")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

fig.suptitle("GAE Raw Throughput Speedup over LBRC nvCOMP Zstd", fontsize=18, fontweight="bold")
fig.savefig(OUT / "gae_raw_over_lbrc_nvcomp_zstd_speedup.png", dpi=220)

print("Wrote", summary)
print("Wrote", OUT / "gae_raw_vs_lbrc_nvcomp_zstd_gbps.png")
print("Wrote", OUT / "gae_raw_vs_lbrc_nvcomp_zstd_cr.png")
print("Wrote", speed_path)
print("Wrote", OUT / "gae_raw_over_lbrc_nvcomp_zstd_speedup.png")
