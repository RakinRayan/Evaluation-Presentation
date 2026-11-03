# Check of the pair 2 preservation result, run after the Kaggle notebook (1 November 2026).
#
# The notebook scores "product" SSIM, DINOv2 and LPIPS on the rectangle around the product. For long, thin products
# (the stylus pen) most of that rectangle is background, so a changed background also lowers the score. This script
# recomputes SSIM on the product pixels only, using the photos and edits the notebook saved, the same product mask and
# the same 512 px size. It needs only numpy, scipy, pandas, pillow and scikit-image (no GPU).
#
# usage: python product_pixel_check.py            (run from this folder)
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
from scipy.stats import wilcoxon
from skimage.metrics import structural_similarity

RES = Path(__file__).resolve().parent / "results"
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)


def product_mask(img):
    # same as the notebook: everything that isn't the white background touching the border
    labels, _ = ndimage.label(np.asarray(img.convert("RGB")).min(axis=2) >= 240)
    border = np.unique(np.r_[labels[0], labels[-1], labels[:, 0], labels[:, -1]])
    return ndimage.binary_fill_holes(ndimage.binary_opening(~np.isin(labels, border[border > 0]), iterations=2))


# the input photo of each product is saved with the rating images
key = pd.read_csv(RES / "rating" / "key.csv")
inputs = {}
for r in key.itertuples():
    inputs.setdefault(r.edit.split("_")[0], RES / "rating" / f"item{r.item:02d}_input.png")

scores = pd.read_csv(RES / "scores.csv")
rows = []
for r in scores.itertuples():
    if r.task == "colour_variant":          # colour edits are meant to change the product
        continue
    photo = Image.open(inputs[r.edit.split("_")[0]]).convert("RGB")
    size = (round(photo.width * 512 / max(photo.size)), round(photo.height * 512 / max(photo.size)))
    folder = "klein" if "klein" in r.system else "qwen"
    inp = photo.resize(size)
    out = Image.open(RES / folder / f"seed{r.seed}" / f"{r.edit}.png").convert("RGB").resize(size)
    mask = product_mask(inp)
    ys, xs = np.where(mask)
    box_area = (xs.max() + 1 - xs.min()) * (ys.max() + 1 - ys.min())
    _, ssim_map = structural_similarity(np.asarray(inp.convert("L")), np.asarray(out.convert("L")), full=True)
    rows.append({"system": r.system, "seed": r.seed, "edit": r.edit, "task": r.task,
                 "ssim_rectangle": r.product_ssim,                 # the notebook's score
                 "ssim_product_pixels": ssim_map[mask].mean(),     # SSIM map averaged over the product pixels only
                 "rectangle_share_of_image": box_area / (size[0] * size[1]),
                 "product_share_of_rectangle": mask.sum() / box_area})
check = pd.DataFrame(rows)
check.to_csv(RES / "product_pixel_ssim.csv", index=False)

print("How much of the rectangle is product:")
print(check.groupby("edit")[["rectangle_share_of_image", "product_share_of_rectangle"]].first().round(2))

per_edit = check.groupby(["edit", "system"])[["ssim_rectangle", "ssim_product_pixels"]].mean().unstack()
print()
print("Mean over the three seeds:")
print(per_edit.round(3))

A, B = "Qwen-Image-2.1", "FLUX.2 klein"
for metric in ["ssim_rectangle", "ssim_product_pixels"]:
    for label, keep in [("all 6 non-colour edits", lambda e: True),
                        ("4 edits where the product should stay in place", lambda e: "lifestyle" not in e)]:
        t = per_edit[metric].loc[[e for e in per_edit.index if keep(e)]]
        d = t[A] - t[B]
        print(f"{metric}, {label}: Qwen {t[A].mean():.3f}, klein {t[B].mean():.3f}, "
              f"klein higher on {(d < 0).sum()} of {len(d)}, Wilcoxon p = {wilcoxon(d).pvalue:.3f}")

sd = check.groupby(["edit", "system"])[["ssim_rectangle", "ssim_product_pixels"]].std().groupby("system").mean()
print()
print("Standard deviation over seeds, mean per edit:")
print(sd.round(3))
