# Redraws the data figures of the report (Figures 2 to 6) from the results folders of the four pairs.
# Figure 1 is a diagram of the framework, not data, so it is not here.
#
# usage: python report_figures.py        (run from this folder; writes the PNG and JPG files to report_figures/)
# needs: matplotlib, numpy, pandas, pillow (no GPU)
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parent
P1 = HERE / "Pair 1 Shelf monitoring" / "results"
P2 = HERE / "Pair 2 Photo editing" / "results"
P3 = HERE / "Pair 3 Super-resolution" / "results"
P4 = HERE / "Pair 4 Image compression" / "results"
OUT = HERE / "report_figures"
OUT.mkdir(exist_ok=True)

A_C, B_C, BASE_C = "#2563EB", "#EA580C", "#8C8C8C"   # System A blue, System B orange, baselines grey
plt.rcParams.update({"font.family": ["Liberation Sans", "DejaVu Sans"], "font.size": 9.5, "axes.titlesize": 10,
                     "axes.labelsize": 9.5, "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold", "savefig.dpi": 220})


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", OUT / name)


def no_axes(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)


# Figure 2: pair 1, F1 and counting error by seed, and recall by how many products are in the photo
seed = pd.read_csv(P1 / "results_by_seed.csv")
photos = pd.read_csv(P1 / "per_photo.csv")
dens = pd.read_csv(P1 / "by_density.csv")
order = ["up to 100", "100 to 200", "over 200"]
photos["density"] = pd.cut(photos["true"], [0, 100, 200, 1000], labels=order)
n_photos = photos[(photos.seed == 0) & (photos.system == "YOLO26s")].density.value_counts().reindex(order)

fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), gridspec_kw={"width_ratios": [1, 1, 1.25]})
x = np.arange(3)
for ax, metric, title, fmt in [(axes[0], "f1", "F1 at the working threshold", "{:.2f}"),
                               (axes[1], "count_error", "Counting error (lower is better)", "{:.1%}")]:
    a = seed[seed.system == "YOLO26s"].sort_values("seed")[metric].to_numpy()
    b = seed[seed.system == "RF-DETR Nano"].sort_values("seed")[metric].to_numpy()
    bars_a = ax.bar(x - 0.19, a, 0.36, color=A_C, label="YOLO26s (A)")
    bars_b = ax.bar(x + 0.19, b, 0.36, color=B_C, label="RF-DETR Nano (B)")
    for bars in (bars_a, bars_b):
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height(), fmt.format(r.get_height()), ha="center", va="bottom", fontsize=7.8)
    ax.set_xticks(x, [f"Seed {s}" for s in range(3)])
    ax.set_title(title)
    if metric == "f1":
        ax.set_ylim(0.70, 0.86)
    else:
        ax.set_ylim(0, 0.17)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
ra = dens[dens.system == "YOLO26s"].set_index("density").loc[order, "recall"].to_numpy()
rb = dens[dens.system == "RF-DETR Nano"].set_index("density").loc[order, "recall"].to_numpy()
ax = axes[2]
for bars in (ax.bar(x - 0.19, ra, 0.36, color=A_C), ax.bar(x + 0.19, rb, 0.36, color=B_C)):
    for r in bars:
        ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.01, f"{r.get_height():.2f}", ha="center", va="bottom", fontsize=7.8)
ax.set_xticks(x, [f"{o}\n({n} photos)" for o, n in zip(order, n_photos.to_numpy())])
ax.set_ylim(0, 1.0)
ax.set_title("Recall by products in the photo")
fig.tight_layout(rect=[0, 0, 1, 0.9])
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=9)
save(fig, "figure2_pair1_results.png")

# Figure 3: pair 2, SSIM on the product pixels per edit (from product_pixel_check.py) and seconds per edit
scores = pd.read_csv(P2 / "scores.csv")
pixels = pd.read_csv(P2 / "product_pixel_ssim.csv")
wide = pixels.groupby(["edit", "system"]).ssim_product_pixels.mean().unstack()
short = {"B00JKQKAYG": "stylus pen", "B07FF6ZGH8": "backpack", "B075XVTSPS": "power bank", "B07VXQXBKV": "USB hub"}
task = pixels.drop_duplicates("edit").set_index("edit").task
names = [f"{short[e.split('_')[0]]}, {task[e].split('_')[0]}" for e in wide.index]
rank = np.argsort(wide["FLUX.2 klein"].to_numpy())
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), gridspec_kw={"width_ratios": [1.6, 1]})
ax = axes[0]
for j, i in enumerate(rank):
    q, k = wide["Qwen-Image-2.1"].iloc[i], wide["FLUX.2 klein"].iloc[i]
    ax.plot([q, k], [j, j], color="#C8C8C8", lw=1.5, zorder=1)
    ax.scatter(q, j, color=A_C, s=28, zorder=2, label="Qwen-Image-2.1 (A)" if j == 0 else None)
    ax.scatter(k, j, color=B_C, s=28, zorder=2, label="FLUX.2 klein (B)" if j == 0 else None)
ax.set_yticks(range(len(rank)), [names[i] for i in rank])
ax.set_xlabel("SSIM on product pixels, mean of 3 seeds (higher = less changed)")
ax.set_title("Product kept intact, 6 non-colour edits")
ax.legend(loc="lower right", frameon=False)
ax.set_xlim(0.0, 1.02)
ax = axes[1]
secs = scores.groupby("system").seconds.median().reindex(["Qwen-Image-2.1", "FLUX.2 klein"])
bars = ax.bar(["Qwen-Image-2.1", "FLUX.2 klein"], secs.to_numpy(), color=[A_C, B_C], width=0.55)
for r in bars:
    ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 2, f"{r.get_height():.0f} s", ha="center", fontsize=9)
ax.set_ylabel("Seconds per edit on a T4 (median)")
ax.set_title("Speed")
ax.set_ylim(0, 135)
fig.tight_layout()
save(fig, "figure3_pair2_preservation_speed.png")

# Figure 4: pair 2, example edits from both models (seed 0)
key2 = pd.read_csv(P2 / "rating" / "key.csv")
rows = [("B00JKQKAYG_studio_background", "Studio background"), ("B075XVTSPS_lifestyle_background", "Lifestyle background"),
        ("B07VXQXBKV_sale_tag", "Sale tag")]
fig, axes = plt.subplots(len(rows), 3, figsize=(4.6, 4.3))
for i, (edit, label) in enumerate(rows):
    item = int(key2[key2.edit == edit].item.iloc[0])
    panels = [("Input", P2 / "rating" / f"item{item:02d}_input.png"),
              ("Qwen-Image-2.1 (A)", P2 / "qwen" / "seed0" / f"{edit}.png"),
              ("FLUX.2 klein (B)", P2 / "klein" / "seed0" / f"{edit}.png")]
    for j, (title, path) in enumerate(panels):
        ax = axes[i][j]
        img = Image.open(path).convert("RGB")
        img.thumbnail((700, 700))
        ax.imshow(img)
        no_axes(ax)
        if i == 0:
            ax.set_title(title, fontweight="bold", fontsize=8.5)
        if j == 0:
            ax.set_ylabel(label, fontsize=8)
fig.tight_layout(pad=0.3, w_pad=0.4, h_pad=0.5)
save(fig, "figure4_pair2_examples.jpg")

# Figure 5: pair 3, crops of three supplier-like inputs next to the real photo (the rating crops, seed 0)
key3 = pd.read_csv(P3 / "rating" / "key.csv")
items = [(4, "Speaker grille"), (7, "Printed marks"), (8, "Label text")]
fig, axes = plt.subplots(len(items), 3, figsize=(6.3, 6.4))
for i, (item, label) in enumerate(items):
    row = key3[key3.item == item].iloc[0]
    letter = {row.X: "X", row.Y: "Y"}
    panels = [("Real photo", f"item{item:02d}_real.png"), ("AuraSR v2 (A)", f"item{item:02d}_{letter['AuraSR v2']}.png"),
              ("HYPIR (B)", f"item{item:02d}_{letter['HYPIR']}.png")]
    for j, (title, name) in enumerate(panels):
        ax = axes[i][j]
        ax.imshow(Image.open(P3 / "rating" / name).convert("RGB"))
        no_axes(ax)
        if i == 0:
            ax.set_title(title, fontweight="bold", fontsize=14)
        if j == 0:
            ax.set_ylabel(label, fontsize=13)
fig.tight_layout()
save(fig, "figure5_pair3_crops.jpg")

# Figure 6: pair 4, rate-distortion curves, with each paper's published Kodak points (Li et al., 2025; Lu et al., 2025)
curves = pd.read_csv(P4 / "curves.csv")
paper = {"HPCM-Base": ([0.1757, 0.2729, 0.4125, 0.5898], [30.7547, 32.4012, 34.2063, 36.0145]),
         "DCAE": ([0.175, 0.292, 0.426, 0.593], [30.74, 32.68, 34.41, 36.11])}
colours = {"HPCM-Base": A_C, "DCAE": B_C, "AVIF": "#7C3AED", "WebP": "#15803D", "JPEG": BASE_C}
labels = {"HPCM-Base": "HPCM-Base (A)", "DCAE": "DCAE (B)", "AVIF": "AVIF", "WebP": "WebP", "JPEG": "JPEG"}
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
for ax, part, title in [(axes[0], "products", "8 product photos"), (axes[1], "kodak", "24 Kodak images")]:
    for codec, colour in colours.items():
        g = curves[(curves.set == part) & (curves.codec == codec)].sort_values("bpp")
        ax.plot(g.bpp, g.psnr, marker="o", ms=2.8, lw=1.4, color=colour, label=labels[codec])
    if part == "kodak":
        for codec in ("HPCM-Base", "DCAE"):
            ax.plot(*paper[codec], ls="none", marker="s", ms=5, mfc="none", mec=colours[codec], mew=1.0, label=f"{codec} paper")
    ax.set_xscale("log")
    ax.set_xlabel("Bits per pixel (log scale)")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title(title)
    ax.grid(alpha=0.25, which="both", lw=0.5)
axes[1].legend(frameon=False, fontsize=8, loc="upper left")
fig.tight_layout()
save(fig, "figure6_pair4_rd_curves.png")
