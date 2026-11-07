# COMP8410 Assignment 1: Evaluation Presentation

Mohammad Rakin Rayan, 48658057, Master of Information Technology (Artificial Intelligence), Macquarie University.

This submission holds the report, presentation, recorded video, code and saved results for my evaluation of four pairs
of image AI models for an online retail business.

| Pair | Task | System A | System B |
|---|---|---|---|
| 1 | Detect and count products in shelf photos (SKU-110K) | YOLO26s | RF-DETR Nano |
| 2 | Edit product photos from a text instruction (ABO) | Qwen-Image-2.1 | FLUX.2 [klein] 4B |
| 3 | Enlarge small supplier photos 4 times | AuraSR v2 | HYPIR |
| 4 | Compress product images for the website | HPCM-Base | DCAE |

## Contents

- `48658057 Report/`: the written report (PDF).
- `48658057 Presentation/`: the slides (PowerPoint), with the recorded narration on each slide.
- `48658057 Video/`: the recorded presentation (MP4), exported from the narrated slides.
- `48658057 Evaluation/`: one folder per pair, each with the notebook I ran on Kaggle (all outputs kept) and a
  `results` folder holding everything the notebook wrote. The folder also has `report_figures.py` (redraws the report
  figures), `statistics_check.py` (Holm correction, bootstrap intervals and effect sizes), `saving_checks.py`
  (uncertainty and sensitivity of the cost figures) and its own README.

See [`48658057 Evaluation/README.md`](48658057%20Evaluation/README.md) for how to re-run each pair, what I implemented
and reused in each one, the data licences and the version history.
