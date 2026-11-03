# COMP8410 Assignment 1: Evaluation code and data

Mohammad Rakin Rayan, 48658057, Master of Information Technology (Artificial Intelligence), Macquarie University.

This folder holds the evaluation behind the report and the presentation. There is one folder per pair. Each one has the
notebook I ran on Kaggle (with all its outputs kept) and a `results` folder with everything the notebook wrote: CSV files,
figures, `run_info.json` and the rating files.

| Folder | Task | System A | System B |
|---|---|---|---|
| Pair 1 Shelf monitoring | Detect and count products in shelf photos | YOLO26s | RF-DETR Nano |
| Pair 2 Photo editing | Edit product photos from a text instruction | Qwen-Image-2.1 | FLUX.2 [klein] 4B |
| Pair 3 Super-resolution | Enlarge small supplier photos 4 times | AuraSR v2 | HYPIR |
| Pair 4 Image compression | Compress product images for the website | HPCM-Base | DCAE |

## How to re-run a pair

1. On Kaggle, create a new notebook and import the `.ipynb` file from the pair's folder.
2. Settings: Accelerator GPU T4 x2, Internet on. For pair 1 also add the SKU-110K dataset as input
   (I used `thedatasith/sku110k-annotations`; the loader reads both the YOLO-label and the original CSV versions).
3. Save Version, then Save & Run All. Everything is written to `/kaggle/working/pairN`, and the last code cell before the
   ratings zips that folder so it can be downloaded.
4. Approximate run times on a T4 (31 October 2026): pair 1 59 min, pair 2 66 min, pair 3 19 min, pair 4 18 min.
5. The very last cell of each notebook (my ratings) is not part of the Kaggle run, so on Kaggle Save & Run All stops there
   with a file-not-found error, which is expected. I ran that cell on Google Colab afterwards. To re-run it, open the
   notebook in Colab, upload `my_ratings.csv` and `key.csv` from `results/rating/` and, for pairs 1 to 3,
   `speed_cost.csv` from `results/` into Colab's `/content` folder, then run only that cell.
6. Pair 4 downloads 6.8 GB of checkpoints from Google Drive. If Drive refuses a download, add the `.pth` files to the
   notebook as a Kaggle dataset; the notebook looks in `/kaggle/input` first. Name each file so that its path contains
   `hpcm` or `dcae` and its file name contains the lambda, for example `hpcm_0.0035.pth`.

Package versions are pinned in the first code cell of each notebook, and the GitHub code used by pairs 3 and 4 is pinned
to a commit. `results/run_info.json` records the date, GPU, PyTorch version, package versions, commits and seeds.

### Report figures

`report_figures.py` in this folder redraws the data figures of the report (Figures 2 to 6) from the `results` folders
of the four pairs: `python report_figures.py` (needs matplotlib, numpy, pandas and pillow; no GPU, a few seconds). It
writes them to `report_figures/`. Figure 1 is a diagram of the framework, not data.

### Extra statistics

`statistics_check.py` in this folder computes, from the saved results only, the Holm-corrected p within each pair,
bootstrap 95% intervals for the mean differences (10,000 resamples, seed 0) and the matched-pairs rank-biserial
correlation for every paired comparison in the report, including the ratings, plus the follow-up checks the report
quotes (shelf-density subgroups and trend, Spearman correlations, per-photo BD-rates against AVIF and the
background check): `python statistics_check.py` (needs
numpy, pandas and scipy; a few seconds). It writes `statistics_check.csv`. Its p-values match the notebooks'.

### My ratings (user feedback)

The unit's update allows the student's own evaluation as user feedback. Each notebook saves the two models' outputs for
the same inputs in `results/rating/` as `itemNN_X` and `itemNN_Y`, in a random order, and writes the key to `key.csv`.
I filled in `my_ratings.csv` (score for X, score for Y, seconds taken) before opening the key. The last cell of each
notebook was added after the Kaggle run: it reads `my_ratings.csv`, `key.csv` and the cost table, then prints the mean
score, the share rated 4 or 5, a Wilcoxon test and my review time. I ran that cell on Google Colab after uploading those
files. In pairs 3 and 4 the first item (a white easel on a white background) cropped to a blank patch, so it is left
blank in `my_ratings.csv` and the cell skips it.

Most timings in `my_ratings.csv` were rounded to 5 s, so the review times the cells print (8.9 s, 17.5 s, 7.1 s and
9.0 s) are rough. The report rounds them to about 9, 18 and 7 s.

Pair 2 includes the output images of all three seeds, because `product_pixel_check.py` needs them. To keep the folder
small, pair 3 includes only the seed 0 images. The scores for every seed are in `scores.csv` and `results_by_seed.csv`,
and re-running a notebook regenerates all of its images.

## Implementation statements

### Pair 1: shelf monitoring (YOLO26s vs RF-DETR Nano)

- **Data:** SKU-110K (Goldman et al., 2019) from a Kaggle copy in YOLO label format. Fixed random split with seed 0:
  500 training, 100 validation and 150 test photos (21,443 products in the test photos).
- **What I implemented:** fine-tuning of both detectors from COCO weights with the same photos, 640 px input, 12 epochs,
  300 detections per photo, seeds 0, 1 and 2; COCO mAP and AP50 with pycocotools; a working threshold picked on the
  validation photos only; per-photo precision, recall, F1 and counting error on the test photos; a paired Wilcoxon test
  over the 150 photos; results by shelf density; time per photo; a simple cost and energy estimate; go/no-go checks; a
  risk table; and the blind rating files.
- **Training recipe:** each library's own recipe at the largest batch that fits a T4: batch 8 for YOLO26s, and batch 4
  with 4 accumulation steps (effective 16) for RF-DETR Nano. So the comparison is between practical recipes, not
  identical ones.
- **Reused:** the `ultralytics` (8.4.189) and `rfdetr` (1.11.2) training code and pretrained weights.
- **Main outputs:** `results_by_seed.csv`, `per_photo.csv`, `paired_test.csv`, `by_density.csv`, `speed_cost.csv`,
  `speed_cost_with_review.csv`, `go_no_go.csv`, `training_time.csv`, `fig_metrics.png`, `fig_example.jpg`.
- **Problem fixed:** the Kaggle copy uses YOLO label files rather than the original CSV, so the loader reads both.
- **Timing note:** the ratings cell prints my time per photo (8.9 s) and the notebook text calls it the manual-check
  time. I was judging the drawn boxes rather than counting products, so the report uses it as the time to review the
  model's boxes, and uses my 50.5 s per photo from the first version for checking a photo by hand (not re-timed).

### Pair 2: photo editing (Qwen-Image-2.1 vs FLUX.2 [klein] 4B)

- **Data:** four catalogue photos from Amazon Berkeley Objects (CC BY 4.0, "Amazon Berkeley Objects, Amazon.com"),
  picked at random with seed 0; the source URL of each photo is in `products.csv`.
- **What I implemented:** 8 edits (two of four tasks per photo: studio background, lifestyle background, colour variant,
  sale tag), each with seeds 0, 1 and 2 and the same photo, instruction and seed for both models; a product mask from the
  white background; CLIP directional similarity, and DINOv2, LPIPS and SSIM on the product only; OCR of the sale tags
  with EasyOCR; paired Wilcoxon tests; seconds per edit; cost per edit; go/no-go checks; a risk table; rating files.
- **Reused:** `diffusers` 0.41.0 pipelines and the model weights. The T4 has no bfloat16, so both models run in float16
  with CPU offload, and Qwen-Image-2.1 also uses 4-bit NF4 quantisation (bitsandbytes) so it fits in 16 GB.
- **Instructions:** written from the ABO product titles, so two products got odd names ("devices pink" for the stylus
  pen and "adapter renewed" for the USB hub). Both models got exactly the same wording.
- **Sale tags:** `sale_tag_ocr.csv` has the OCR text of every tag. Exact matches: Qwen-Image-2.1 1 of 6, klein 2 of 6.
  Tags containing SALE 20% OFF: Qwen-Image-2.1 2 of 6, klein 3 of 6.
- **Main outputs:** `scores.csv`, `results_by_seed.csv`, `paired_test.csv`, `by_task.csv`, `sale_tag_ocr.csv`,
  `speed_cost.csv`, `speed_cost_with_review.csv`, `go_no_go.csv`, `fig_metrics.png`, `fig_examples.jpg`, and the
  edited images in `klein/` and `qwen/`.
- **Problem fixed:** the first run stalled while loading Qwen-Image-2.1 with no output, so each model got its own cell
  with a progress line per edit and a CSV saved after each model; the re-run finished in 66 minutes.
- **Check added after the run (1 November 2026):** the notebook finds the product from the white background, but its
  "product" SSIM, DINOv2 and LPIPS are computed on the rectangle around the product. For the stylus pen that rectangle
  covers 96% of the image and only 17% of it is pen, so a new background also lowers the score.
  `product_pixel_check.py` recomputes SSIM on the product pixels only, from the images the notebook saved, with the same
  mask and the same 512 px size. Run it from this folder with `python product_pixel_check.py` (needs numpy, scipy,
  pandas, pillow and scikit-image; no GPU, under a minute). It writes `results/product_pixel_ssim.csv` and prints the
  comparisons used in the report: on product pixels klein is higher on 5 of 6 edits (0.717 against 0.635, p = .31) and
  on 4 of 4 in-place edits, the studio backgrounds and sale tags (0.948 against 0.773, p = .125). I did not change the
  notebook; the report gives both the rectangle and the product-pixel results.

### Pair 3: super-resolution (AuraSR v2 vs HYPIR)

- **Data:** eight Amazon Berkeley Objects photos at 1024 px, shrunk 4 times in two ways (clean bicubic, and supplier-like
  with a 0.8 px blur and JPEG quality 60), giving 16 inputs with a known real photo.
- **What I implemented:** enlargement with both models and a bicubic baseline, seeds 0, 1 and 2; PSNR and SSIM on luma,
  LPIPS, MUSIQ (pyiqa 0.1.16), CIEDE2000 colour error on the product, and OCR word recall and invented words; results by
  type of damage; paired Wilcoxon tests; seconds per photo; cost per photo; go/no-go checks; a risk table; rating files.
- **Reused:** `aura-sr` 0.0.4 and the HYPIR code at commit b61d107 with the authors' test settings. Stable Diffusion 2.1
  is no longer on its official Hugging Face page, so the notebook loads a community copy pinned to a fixed revision.
- **Settings:** HYPIR runs in float32 because float16 gave black images on the T4, which makes it slower. Both models
  turned out to be effectively deterministic, so the three seeds agree within 0.01 dB on the mean (0.02 dB per input).
- **Text check:** OCR is only run on the six inputs whose real photo has readable text. Even bicubic gets 0.67 invented
  words per photo from OCR misreading blur, so the invented-word count has a noise floor.
- **Main outputs:** `scores.csv`, `results_by_seed.csv`, `paired_test.csv`, `by_damage.csv`, `ocr_summary.csv`,
  `speed_cost.csv`, `speed_cost_with_check.csv`, `go_no_go.csv`, `fig_metrics.png`, `fig_crops.jpg`, and the
  enlargements in `aurasr/`, `hypir/` and `bicubic/`.

### Pair 4: learned image compression (HPCM-Base vs DCAE)

- **Data:** eight Amazon Berkeley Objects product photos (cropped to multiples of 256 px) and the 24 Kodak images.
- **What I implemented:** encoding and decoding with the four MSE checkpoints of each codec (the authors' code at commits
  c271cfe and e2525a0, checkpoints fingerprinted in `checkpoints.csv`) and with JPEG, WebP and AVIF at nine quality
  levels, all written as real byte strings; bits per pixel, PSNR and MS-SSIM; BD-rate with a piecewise cubic fit; a check
  of my Kodak curves against each paper's own curve; a per-image BD-rate with a Wilcoxon test; encode and decode time; a
  test that decodes a GPU-encoded file on a CPU; CDN cost at today's quality; go/no-go checks; a risk table; rating files.
- **Reused:** the authors' model and entropy-coding code, CompressAI 1.2.8 and imagecodecs 2026.1.14.
- **Main outputs:** `rd_points.csv`, `curves.csv`, `bd_rates.csv`, `bd_per_image.csv`, `paired_test.csv`,
  `cross_device_decode.csv`, `speed_cost.csv`, `go_no_go.csv`, `fig_rd_curves.png`, `fig_crops.jpg`, and the decoded
  images in `decoded/`.
- **Problems fixed before the GPU run:** WebP defaulted to lossless, the decode timer included the metric time, and
  repeated quality points broke the cubic fit; all three are fixed in the notebook.
- **Notes:** `bd_rates.csv` has the pooled BD-rates at equal PSNR and at equal MS-SSIM, and the report gives all of them.
  The cross-device test decodes on a CPU a file encoded on the GPU; there is no CPU-encode, CPU-decode control.

## Version history

- **Version 1 (26 October 2026, not submitted):** a much larger evaluation that went well beyond the unit's 40-hour guide.
- **Version 2 (31 October 2026):** re-scoped to one shorter notebook per pair with the same models; every pair was re-run
  on Kaggle. Only my 50.5 s manual shelf-check timing comes from version 1.
- **Version 3 (1 November 2026):** corrections after re-checking the results. Pair 2 was re-scored on product pixels with
  `product_pixel_check.py` (all seeds' images added for this), all pair 4 BD-rates are reported, the review timings are
  described as rough, `report_figures.py` was added, and the report and slides were corrected where they overstated
  small or confounded results. A second check the same day corrected how the densest shelf photo's detections were
  described (boxes drawn against products matched) and added the colour-edit results. A last pass added
  `statistics_check.py` (Holm correction, bootstrap intervals and effect sizes) and the report text that uses it. No
  notebook code or output was changed.
- **Readability pass (3 November 2026):** I added explanatory comment lines to the code cells and short markdown
  headings (Setup, Figures, Run record, My ratings: results) to each notebook, plus a note above the Kaggle cell that
  prints "ratings not filled in yet", pointing to the last cell where the ratings are. Nothing was re-run: every line
  of code and every output is exactly as it was, only comment lines and text cells were added.

## Data and licences

- SKU-110K: research use only (Goldman et al., 2019), loaded from a Kaggle copy.
- Amazon Berkeley Objects: CC BY 4.0, "Amazon Berkeley Objects, Amazon.com" (Collins et al., 2022).
- Kodak Lossless True Color Image Suite (Franzen, n.d.).
- Model licences are listed in each notebook's risk table and in the report. No personal data was collected.
