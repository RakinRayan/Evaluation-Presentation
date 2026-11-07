# Uncertainty and sensitivity of the report's cost figures, computed from the saved results only.
# No GPU, under a second. Run from the 48658057 Evaluation folder:  python saving_checks.py
# Needs numpy and pandas.
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
P1 = HERE / "Pair 1 Shelf monitoring" / "results"
P2 = HERE / "Pair 2 Photo editing" / "results"
P3 = HERE / "Pair 3 Super-resolution" / "results"


def wilson(k, n, z=1.96):
    """95% Wilson interval for a proportion k/n."""
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def my_scores(folder, system):
    """My blind scores for one system (items left blank are skipped)."""
    r = pd.read_csv(folder / "rating" / "my_ratings.csv").merge(pd.read_csv(folder / "rating" / "key.csv"))
    r = r.dropna(subset=["score_X", "score_Y"])
    return np.where(r.X == system, r.score_X, r.score_Y)


# Pair 1: the shelf saving per store per day, for different manual check times.
# The 50.5 s manual time comes from version 1 and has no log; the review time is the mean of my_ratings.csv.
review_s = pd.read_csv(P1 / "rating" / "my_ratings.csv").seconds_to_check.mean()
PHOTOS, RATE = 595, 31.15                     # photos per store per day; Retail Award level 1 + 12% super (AUD/h)
print(f"pair 1: review time {review_s:.1f} s per photo")
for manual_s in [50.5, 40, 25, review_s]:
    saving = PHOTOS * (manual_s - review_s) / 3600 * RATE
    print(f"pair 1: manual check {manual_s:.1f} s -> saving AUD {saving:.0f} per store per day")

# Pairs 2 and 3: share of outputs I rated 4 or 5 (publishable), its 95% Wilson interval, and the monthly saving.
# "notebook" = the formula in the ratings cell: volume x share x (price - GPU cost - review cost).
# "every attempt charged" = volume x share x price - volume x (GPU cost + review cost): the GPU and review time of
# rejected outputs is also paid, and rejected outputs go to the agency or a reshoot as before.
cases = [("pair 2", "FLUX.2 klein", P2, "speed_cost_with_review.csv", "review AUD per edit", "GPU AUD per edit", 2000, 17.32),
         ("pair 2", "Qwen-Image-2.1", P2, "speed_cost_with_review.csv", "review AUD per edit", "GPU AUD per edit", 2000, 17.32),
         ("pair 3", "HYPIR", P3, "speed_cost_with_check.csv", "check AUD per photo", "GPU AUD per photo", 1500, 36.09)]
for pair, system, folder, file, review_col, gpu_col, volume, price in cases:
    cost = pd.read_csv(folder / file).set_index("system").loc[system]
    per_attempt = cost[gpu_col] + cost[review_col]
    s = my_scores(folder, system)
    good, n = int((s >= 4).sum()), len(s)
    lo, hi = wilson(good, n)
    notebook = volume * good / n * (price - per_attempt)
    charged = lambda share: volume * share * price - volume * per_attempt
    print(f"{pair} {system}: {good} of {n} publishable, 95% interval {lo:.2f} to {hi:.2f}; "
          f"saving AUD {notebook:,.0f} (notebook), AUD {charged(good / n):,.0f} with every attempt charged, "
          f"range AUD {charged(lo):,.0f} to {charged(hi):,.0f}")

lo, hi = wilson(100, 200)
print(f"editing pilot: 100 of 200 edits approved would give a 95% interval of {lo:.3f} to {hi:.3f}")

# Pair 1: products an unchecked count would miss on an average test photo (all seeds).
pp = pd.read_csv(P1 / "per_photo.csv")
rf = pp[pp.system == "RF-DETR Nano"]
print(f"pair 1: RF-DETR Nano misses {(rf.true * (1 - rf.recall)).mean():.1f} products per test photo on average")

# Pair 3: HYPIR saving once the automatic gate in report Table 14 is applied to the seven rated photos (seed 0).
# The gate passes an enlargement only if OCR finds no more invented words than on bicubic (photos with text) AND its
# colour error is no more than 1.0 above bicubic's (every photo). Enlargements that fail go to a reshoot instead.
sc = pd.read_csv(P3 / "scores.csv")
s0 = sc[sc.seed == 0]
bic = s0[s0.system == "Bicubic"].set_index("job")
hyp_s = s0[s0.system == "HYPIR"].set_index("job")
k3 = pd.read_csv(P3 / "rating" / "my_ratings.csv").merge(pd.read_csv(P3 / "rating" / "key.csv")).dropna(subset=["score_X", "score_Y"])
hyp = np.where(k3.X == "HYPIR", k3.score_X, k3.score_Y)
jobs = k3.input
words_ok = [pd.isna(hyp_s.loc[j, "invented_words"]) or hyp_s.loc[j, "invented_words"] <= bic.loc[j, "invented_words"] for j in jobs]
colour_ok = [hyp_s.loc[j, "delta_e"] - bic.loc[j, "delta_e"] <= 1.0 for j in jobs]
usable = int(((hyp >= 4) & np.array(words_ok) & np.array(colour_ok)).sum())
cost = pd.read_csv(P3 / "speed_cost_with_check.csv").set_index("system").loc["HYPIR"]
saving = 1500 * usable / len(k3) * (36.09 - cost["GPU AUD per photo"] - cost["check AUD per photo"])
print(f"pair 3 HYPIR after the text and colour gate: {usable} of {len(k3)} rated photos usable "
      f"(rated 4 or 5 and passing the gate), saving AUD {saving:,.0f} a month")
