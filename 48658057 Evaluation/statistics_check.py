# Extra statistics for the report, computed only from the saved results (no GPU, a few seconds).
#
# For every paired comparison in the report: the mean difference A minus B with a bootstrap 95% interval,
# the Wilcoxon p (same test as the notebooks), Holm-adjusted p within each pair, and the matched-pairs
# rank-biserial correlation as an effect size (+1 = A higher on every item, -1 = B higher on every item).
#
# usage: python statistics_check.py      (run from this folder; writes statistics_check.csv)
# needs: numpy, pandas, scipy
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon

HERE = Path(__file__).resolve().parent
P1 = HERE / "Pair 1 Shelf monitoring" / "results"
P2 = HERE / "Pair 2 Photo editing" / "results"
P3 = HERE / "Pair 3 Super-resolution" / "results"
P4 = HERE / "Pair 4 Image compression" / "results"
rng = np.random.default_rng(0)


def describe(pair, metric, a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    boots = rng.choice(d, size=(10000, len(d)), replace=True).mean(axis=1)
    nz = d[d != 0]
    ranks = rankdata(np.abs(nz))
    r = (ranks[nz > 0].sum() - ranks[nz < 0].sum()) / ranks.sum() if len(nz) else 0.0
    p = wilcoxon(d).pvalue if len(nz) else 1.0
    return {"pair": pair, "metric": metric, "items": len(d), "mean A": a.mean(), "mean B": b.mean(),
            "A minus B": d.mean(), "CI low": np.percentile(boots, 2.5), "CI high": np.percentile(boots, 97.5),
            "Wilcoxon p": p, "rank-biserial r": r}


def ratings(folder, a_name, b_name):
    r = pd.read_csv(folder / "rating" / "my_ratings.csv").merge(pd.read_csv(folder / "rating" / "key.csv"))
    r = r.dropna(subset=["score_X", "score_Y"])
    a = np.where(r.X == a_name, r.score_X, r.score_Y)
    b = np.where(r.X == b_name, r.score_X, r.score_Y)
    return a, b


rows = []

# pair 1: per photo, averaged over the three seeds (A = YOLO26s, B = RF-DETR Nano)
pp = pd.read_csv(P1 / "per_photo.csv").groupby(["photo", "system"])[["precision", "recall", "f1", "count_error"]].mean().unstack()
for m in ["precision", "recall", "f1", "count_error"]:
    rows.append(describe(1, m, pp[m]["YOLO26s"], pp[m]["RF-DETR Nano"]))
a, b = ratings(P1, "YOLO26s", "RF-DETR Nano")
rows.append(describe(1, "my rating", a, b))

# pair 2: per edit, averaged over the three seeds (A = Qwen-Image-2.1, B = FLUX.2 klein)
sc = pd.read_csv(P2 / "scores.csv").groupby(["edit", "task", "system"])[["clip_dir", "product_ssim"]].mean().unstack()
rows.append(describe(2, "clip_dir (8 edits)", sc["clip_dir"]["Qwen-Image-2.1"], sc["clip_dir"]["FLUX.2 klein"]))
keep = sc[sc.index.get_level_values("task") != "colour_variant"]
rows.append(describe(2, "SSIM rectangle (6 edits)", keep["product_ssim"]["Qwen-Image-2.1"], keep["product_ssim"]["FLUX.2 klein"]))
px = pd.read_csv(P2 / "product_pixel_ssim.csv").groupby(["edit", "task", "system"]).ssim_product_pixels.mean().unstack()
rows.append(describe(2, "SSIM product pixels (6 edits)", px["Qwen-Image-2.1"], px["FLUX.2 klein"]))
inplace = px[px.index.get_level_values("task").isin(["studio_background", "sale_tag"])]
rows.append(describe(2, "SSIM product pixels (4 in-place edits)", inplace["Qwen-Image-2.1"], inplace["FLUX.2 klein"]))
a, b = ratings(P2, "Qwen-Image-2.1", "FLUX.2 klein")
rows.append(describe(2, "my rating", a, b))

# pair 3: per input, averaged over the three seeds (A = AuraSR v2, B = HYPIR)
s3 = pd.read_csv(P3 / "scores.csv")
s3 = s3[s3.system != "Bicubic"].groupby(["job", "system"])[["psnr_y", "ssim_y", "lpips", "musiq", "delta_e", "word_recall", "invented_words"]].mean().unstack()
for m in ["psnr_y", "ssim_y", "lpips", "musiq", "delta_e"]:
    rows.append(describe(3, m, s3[m]["AuraSR v2"], s3[m]["HYPIR"]))
text = s3.dropna(subset=[("word_recall", "AuraSR v2")])
for m in ["word_recall", "invented_words"]:
    rows.append(describe(3, m, text[m]["AuraSR v2"], text[m]["HYPIR"]))
a, b = ratings(P3, "AuraSR v2", "HYPIR")
rows.append(describe(3, "my rating", a, b))

# pair 4: per-image BD-rate of HPCM-Base against DCAE, tested against zero (negative = HPCM-Base needs fewer bits)
bd = pd.read_csv(P4 / "bd_per_image.csv")
for s in ["kodak", "products"]:
    v = bd[bd.set == s]["BD A vs B %"].to_numpy()
    rows.append(describe(4, f"per-image BD-rate % ({s})", v, np.zeros_like(v)))
a, b = ratings(P4, "HPCM-Base", "DCAE")
rows.append(describe(4, "my rating", a, b))

main = len(rows)   # the rows above are the report's main tests; the rows below are follow-up checks, left out of Holm

# follow-up checks quoted in the report (not part of the Holm families)
from scipy.interpolate import PchipInterpolator
from scipy.stats import spearmanr
import math

# pair 1: recall by shelf density, per photo averaged over seeds
dens = pd.read_csv(P1 / "per_photo.csv").groupby(["photo", "system"])[["true", "recall"]].mean().unstack()
true = dens["true"]["YOLO26s"]
for label, mask in [("recall, over 200 products", true > 200), ("recall, up to 100 products", true <= 100)]:
    rows.append(describe(1, label, dens["recall"]["YOLO26s"][mask], dens["recall"]["RF-DETR Nano"][mask]))


def spearman(pair, label, x, y):
    rho, p = spearmanr(x, y)
    return {"pair": pair, "metric": label, "items": len(x), "rho": rho, "Wilcoxon p": np.nan, "rho p": p}


# pair 1: my score against the seed 0 F1 of the same photo
pp0 = pd.read_csv(P1 / "per_photo.csv")
pp0 = pp0[pp0.seed == 0].set_index(["system", "photo"]).f1
kr = pd.read_csv(P1 / "rating" / "key.csv").merge(pd.read_csv(P1 / "rating" / "my_ratings.csv"))
sc1 = [(x.score_X, pp0[(x.X, x.photo)]) for x in kr.itertuples()] + [(x.score_Y, pp0[(x.Y, x.photo)]) for x in kr.itertuples()]
rows.append(spearman(1, "my rating vs seed 0 F1", [a for a, _ in sc1], [b for _, b in sc1]))

# pair 3: my score against the metrics of the rated output
s3all = pd.read_csv(P3 / "scores.csv").groupby(["system", "job"]).mean(numeric_only=True)
kr3 = pd.read_csv(P3 / "rating" / "key.csv").merge(pd.read_csv(P3 / "rating" / "my_ratings.csv")).dropna(subset=["score_X", "score_Y"])
pts = [(x.score_X, s3all.loc[(x.X, x.input)]) for x in kr3.itertuples()] + [(x.score_Y, s3all.loc[(x.Y, x.input)]) for x in kr3.itertuples()]
for m in ["musiq", "lpips", "delta_e"]:
    rows.append(spearman(3, f"my rating vs {m}", [a for a, _ in pts], [b[m] for _, b in pts]))


# pair 4: per-image BD-rate against AVIF, with the notebook's method (Pchip on log rate, PSNR)
def fit(bpp, q):
    o = np.argsort(q)
    q, r = np.asarray(q, float)[o], np.log(np.asarray(bpp, float)[o])
    keep = np.r_[True, np.diff(q) > 1e-9]
    return q[keep], r[keep]


def bd_rate(anchor_bpp, anchor_q, test_bpp, test_q):
    qa, ra = fit(anchor_bpp, anchor_q)
    qt, rt = fit(test_bpp, test_q)
    lo, hi = max(qa.min(), qt.min()), min(qa.max(), qt.max())
    gap = (PchipInterpolator(qt, rt).integrate(lo, hi) - PchipInterpolator(qa, ra).integrate(lo, hi)) / (hi - lo)
    return (math.exp(gap) - 1) * 100


rd = pd.read_csv(P4 / "rd_points.csv")
rd = rd[rd.set == "products"]
for codec in ["HPCM-Base", "DCAE"]:
    v = []
    for image, g in rd.groupby("image"):
        avif, test = g[g.codec == "AVIF"], g[g.codec == codec]
        v.append(bd_rate(avif.bpp, avif.psnr, test.bpp, test.psnr))
    v = np.array(v)
    row = describe(4, f"BD-rate % vs AVIF per product photo ({codec})", v, np.zeros_like(v))
    row["photos beating AVIF by 10%"] = int((v <= -10).sum())
    rows.append(row)
    # does a plainer background mean a bigger saving? share of near-white pixels (all channels above 245),
    # measured on the decoded image at lambda 0.0067, against the per-photo BD-rate
    from PIL import Image
    white = [(np.asarray(Image.open(P4 / "decoded" / f"dcae_{image}.png").convert("RGB")) > 245).all(axis=2).mean()
             for image, _ in rd.groupby("image")]
    rows.append(spearman(4, f"white background share vs BD-rate vs AVIF ({codec})", white, v))

# pair 1: does under-counting grow with shelf density? signed bias (found - true) / true against the true count
bias = pd.read_csv(P1 / "per_photo.csv").groupby(["photo", "system"])[["true", "found"]].mean().reset_index()
bias["bias"] = (bias.found - bias.true) / bias.true
for system, g in bias.groupby("system"):
    rows.append(spearman(1, f"true count vs signed counting bias ({system})", g.true, g.bias))

out = pd.DataFrame(rows)
# Holm correction within each pair, over that pair's main tests (ratings included)
out["Holm p"] = np.nan
for pair, g in out.iloc[:main].groupby("pair"):
    order = g["Wilcoxon p"].sort_values().index
    m = len(order)
    adj, running = [], 0.0
    for k, i in enumerate(order):
        running = max(running, min(1.0, (m - k) * out.at[i, "Wilcoxon p"]))
        adj.append((i, running))
    for i, v in adj:
        out.at[i, "Holm p"] = v
out.to_csv(HERE / "statistics_check.csv", index=False)
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", None)
print(out.round(4).to_string(index=False))
