# Project Report — Visual Defect Inspection

<!-- STATUS:START -->
**Status of this document.** Design, code and reasoning are complete. The authoring environment had no internet, no PyTorch and no Docker, so the dataset was **not downloaded, the model was not trained, and the API/Docker were not executed there**. Every sentence marked **PENDING** must be filled from `reports/metrics.json`, `data/processed/audit.json` and `reports/training_history.json` after running the pipeline. No result in this document is invented.
<!-- STATUS:END -->

## 1. Executive Summary
A manufacturing client wants automatic *normal vs defective* classification of product images. The original assessment dataset was not provided, therefore **a publicly available industrial defect dataset (KolektorSDD2) was selected to demonstrate the complete end-to-end solution.** It is not the company's private data.

Solution: audit → leakage-safe split → EfficientNet-B0 (ImageNet) fine-tuned in two phases with class-weighted loss → threshold tuned on validation for high defect recall → FastAPI service in Docker.

Key decisions (each justified below):

| Decision | Choice | Short reason |
|---|---|---|
| Dataset | KolektorSDD2 | real production defects, binary-convertible, enough positives for trustworthy test metrics |
| Input size | 576×208 (≈ native) | defects are tiny; a 224×224 squash would erase them |
| Model | EfficientNet-B0 | small (~5.3M params), accurate transfer learner, fast on CPU |
| Imbalance | class-weighted CE only | one simple technique; sampler would double-count |
| Checkpoint metric | validation PR-AUC | threshold-free, appropriate under imbalance |
| Threshold | max F2 on validation | missed defects cost more than false alarms |
| Serving | FastAPI, model loaded once, CPU Docker | simple, realistic, maintainable |

## 2. Problem Definition
Binary classification, positive class = **defective**.
- **False Negative (FN)** = defective predicted normal → a faulty product ships. Usually the costly error (returns, safety, reputation).
- **False Positive (FP)** = normal predicted defective → unnecessary scrap/re-inspection. Cheaper, but too many erodes trust in the system.

So accuracy alone is not the target: with ~1 defective per 8–9 normals, "always normal" already scores ≈ 89% accuracy and catches nothing. Primary metrics: **recall, precision, F1/F2, PR-AUC, confusion matrix.**

## 3. Dataset Analysis
**Selection (compared against the assessment needs: real defects, normal/defective labels, transfer-learning size, public, CPU-trainable):**

| Dataset | Verdict |
|---|---|
| NEU Surface Defect | 6 defect classes, **no normal class** (from memory, not re-verified) → cannot make normal-vs-defective without inventing a normal class. Rejected. |
| DAGM 2007 | Defects are **artificially generated** (verified) → less realistic. Rejected. |
| MVTec AD | Excellent benchmark but 15 unrelated object categories, ~GB-scale, built for *unsupervised* use (train set is all-good). Mixing categories is not the assessment's scenario. Not chosen. |
| KolektorSDD (v1) | Real, 399 images (52 defective). Too few positives: a 15% test split holds ≈ 8 defects → metrics too noisy. Supported by the code, not the default. |
| **KolektorSDD2 (chosen)** | Real production surfaces; **356 defective / 2,979 normal**; ~230×630 px; official **train 246+/2085−, test 110+/894−**; several defect types (scratches, spots, imperfections); CC BY-NC-SA 4.0 (verified from the dataset page). 110 test positives gives usable recall estimates. |

**Measured statistics** (auto-filled from `data/processed/audit.json` by `src.fill_results`; expected from published counts: imbalance ≈ 8.4 normal : 1 defective):

<!-- AUDIT:START -->
PENDING — run the pipeline.
<!-- AUDIT:END -->

## 4. Data Preparation
`python -m src.prepare_data`:
1. Pair each image with its segmentation mask (`<id>_GT.png`).
2. **Mapping: mask all-zero → Normal (0); any non-zero pixel → Defective (1).** (Multiple defect types, if any, all map to Defective.)
3. Check every file decodes; record sizes; MD5 duplicate groups; defect area fraction over train+val only (a sanity check on defect size; the input resolution was set beforehand from the published image size — the authors use 232×640 W×H, portrait — and never from test images).
4. Split (Section 7), then remove any train/val image whose **bytes are identical** to an image in a later split (test is never modified). `audit.json` records how many were removed per class and how many had conflicting labels; the script prints a WARNING if any *defective* image was removed, so legitimate positives are never dropped silently. Masks are read from raw pixel values (not colour-converted), so palette/1-bit/RGB masks map identically.

## 5. Data Augmentation
Train only; val/test get Resize + Normalize only.

| Transform | Why it is realistic |
|---|---|
| Horizontal & vertical flip | scratches/spots have no preferred orientation |
| Rotation ±5°, translate ≤3%, scale 0.95–1.05 | small pose variation on the line |
| Brightness/contrast ±20% | lighting drift |
| Gaussian blur (p=0.1, mild) | focus variation |
| **No random crop / cutout** | could remove the tiny defect while the label stays "defective" |

## 6. Class Imbalance Handling
Distribution (measured values: Section 3 audit block; published ≈ 8.5:1 in train) → moderate imbalance → **class-weighted CrossEntropy** (`w_c = N / (2·n_c)`, computed from the training split and logged). Reason: simplest technique that directly raises the cost of missing defects, no change to data pipeline. A WeightedRandomSampler was deliberately **not** added on top (double compensation distorts probabilities). Imbalance is also handled at evaluation (PR-AUC, recall/precision instead of accuracy) and decision time (threshold).

## 7. Train / Validation / Test Strategy
KolektorSDD2 ships an **official train/test split**; the test set is kept as-is (comparable to the literature, and it is never touched during development). Validation = stratified 15% of official train (≈350 images), seed 42 → from the published counts this gives ≈ 1,980 train / 350 val / 1,004 test, i.e. ≈ 59% / 10.5% / 30% of the data (exact numbers: `audit.json`), instead of the textbook 70/15/15. Reason: respecting the official test split is more valuable than hitting 70/15/15.
**Test-set discipline:** the test split is read only by `evaluate.py` for the final report. Nothing in `train.py` touches it; checkpoint (best validation PR-AUC), threshold (validation F2) and all hyper-parameters (fixed defaults, none tuned) are decided without it. During development use `python -m src.evaluate --split val` (writes to `reports/val/`, `examples/val/`, so it can never overwrite final test artefacts); run the test evaluation once at the end, and if anything is changed afterwards, state that openly in the report.
Leakage controls: fixed seed; split written to `splits.csv`; exact-duplicate check across splits; threshold and checkpoint chosen on validation only; test used once for reporting. Residual risk: KSDD2 has no item IDs, so near-identical shots of the same item could exist across train/val (stated in Limitations). For KolektorSDD (v1) the code instead uses `StratifiedGroupKFold` grouped by physical item, because its 8 surfaces per item would otherwise leak.

## 8. Model Selection
Candidates: ResNet18 (11.7M params, robust, larger), MobileNetV3 (very fast, usually lower accuracy on fine textures), **EfficientNet-B0 (≈5.3M params with the 1000-class head, ≈4.0M with our 2-class head; strong ImageNet features; ~0.4 GFLOPs @224)**, B1+ (slower, little expected gain at this data scale). Final choice should be defended with the measured numbers: if Section 20 shows latency/accuracy is unsatisfactory, swap `build_model` (single function) and re-run; ResNet18 is the first alternative.

## 9. Why the Selected Model
- Pretrained on ImageNet → edges/textures transfer to surface defects.
- Small enough for CPU serving and a small Docker image; no GPU required.
- Fully convolutional → accepts the non-square 576×208 input unchanged.
- Gives a usable Grad-CAM layer for error analysis.

**Why transfer learning, not scratch:** ≈1,980 training images (only ≈210 defective, after the validation carve-out) is far too little to learn good visual features from scratch; pretrained features reduce overfitting and training time. **Why not a huge model:** more latency, memory and overfit risk for a binary task on a few thousand images.

## 10. Training Methodology
`python -m src.train`: AdamW, batch 16, weight decay 1e-4, cosine LR, seed 42, deterministic cudnn.
1. **Phase 1 (3 epochs, lr 1e-3):** new 2-class head (dropout 0.3) on a frozen backbone — protects pretrained weights from a random head's large gradients.
2. **Phase 2 (≤12 epochs, lr 1e-4):** unfreeze feature blocks 5–8; lower blocks stay frozen with BatchNorm statistics fixed.
3. Early stopping (patience 5); keep the epoch with best **validation PR-AUC**.
4. Pick the threshold on validation: maximise **F2** (recall weighted 4× precision), ties → higher threshold.

**Training run** (auto-filled from `reports/training_history.json`):

<!-- TRAIN:START -->
PENDING — run the pipeline.
<!-- TRAIN:END -->

Note: reproducibility is "practical", not bit-exact (GPU kernels, library versions).

## 11. Evaluation Metrics
Accuracy (context only), **precision, recall, F1, F2**, specificity, TP/TN/FP/FN, confusion matrix, PR-AUC, ROC-AUC, confidence by outcome, latency, model size. Reported at two operating points: default 0.5 and the validation-tuned threshold, so the effect of tuning is visible and not hidden.

## 12. Evaluation Results
Auto-filled from `reports/metrics.json` (test split, evaluated once):

<!-- RESULTS:START -->
PENDING — run the pipeline.
<!-- RESULTS:END -->

Report honestly if recall at the tuned threshold is below what a factory would accept; with ~110 test positives the confidence interval on recall is still several points wide.

## 13. Confusion Matrix
`reports/confusion_matrix.png` (two panels: 0.5 and tuned threshold).

<!-- CM:START -->
PENDING — run the pipeline.
<!-- CM:END -->

## 14. False Positive Analysis — **PENDING (needs real images)**
Open `examples/false_positives/` (sorted by highest P(defective); each figure shows the input, Grad-CAM and the mask). For each, write only what is visible: Is Grad-CAM on a real blemish, texture, edge or reflection? Is the image darker/brighter than typical? Is the ground-truth mask possibly missing a faint mark (label noise)? Do **not** list causes you did not see.

## 15. False Negative Analysis — **PENDING (needs real images)**
Open `examples/false_negatives/` (lowest P(defective) first). Compare Grad-CAM with the green/red ground-truth region: is the defect very small (see `defect_area_fraction` in audit.json), low contrast, thin, at the border, or does the model look elsewhere? Report the ratio of tiny vs larger defects among misses.

## 16. Error / Failure Analysis
Template to complete with evidence: (a) `confidence_by_outcome` in metrics.json — are FNs near the threshold (recoverable by human review) or confidently wrong (harder problem)? (b) share of errors tied to tiny defects → candidate fix: higher resolution or tiling. (c) share linked to lighting/contrast → candidate fix: stronger photometric augmentation.
Production failure scenarios (not testable on this dataset): **new defect type** (supervised model may call it normal), **camera/lens change**, **lighting change**, **new product variant**, **distribution shift** over time. Mitigation: monitoring (Section 22) and a human-review path.

## 17. Production Inference Architecture
Client → FastAPI → validate (type, size, decode, min side) → preprocess (same `eval_transform` as evaluation → no train/serve skew) → EfficientNet (loaded once, `inference_mode`) → softmax → threshold → JSON. The checkpoint stores weights, threshold, input size and class names, so the service cannot drift out of sync with training. Weights are loaded with `torch.load(weights_only=True)` (no pickle code execution).

## 18. API Implementation
`GET /health` → `{"status":"healthy","model_loaded":true}` (503 if the model failed to load). `POST /predict` (multipart `file`) → `predicted_class`, `confidence`, `defect_probability`, `threshold`, `review_recommended`, `latency_ms`.
- `confidence` = probability of the predicted class. With a tuned threshold < 0.5 a "defective" call can have confidence < 0.5; such borderline cases carry `review_recommended=true` (|p − threshold| < 0.15, env `REVIEW_MARGIN`). Probabilities are **not calibrated**.
- Status codes: 400 empty/corrupt/too small, 413 > `MAX_UPLOAD_MB` (10), 415 non-image content type, 500 inference error, 503 model not loaded.
- Validation is by actually decoding with Pillow (content-type headers can lie); decompression-bomb cap 50 MP.
- Logging: startup, model load time/device, each prediction (file, size, class, confidence, latency), rejected inputs, exceptions.
- Tests (`tests/`): health, valid prediction, corrupt image, empty/wrong type, missing model → 503, metric maths. API tests use a random-weight checkpoint, so they verify the service, not accuracy. **Not executed in the authoring environment (no torch) — run `python -m pytest -q`.** The torch-free metric tests were run manually and passed; the data-split script was exercised on synthetic folders (plumbing only) and verified no item crosses splits.

## 19. Docker and Deployment
`docker build -t defect-api .` / `docker run --rm -p 8000:8000 defect-api`. Python 3.11-slim, CPU-only torch wheels, only inference files copied, non-root user, `HEALTHCHECK` on `/health`. Model is baked in (immutable, versioned image) and can be overridden with a volume + `MODEL_PATH`. **Docker build was not executed in the authoring environment; PENDING verification.** Scale-out: stateless container → run several replicas behind a load balancer.

## 20. Accuracy / Latency / Model Trade-offs
**Measured** by `evaluate.py` and auto-filled:

<!-- LATENCY:START -->
PENDING — run the pipeline.
<!-- LATENCY:END -->
Reasoning to keep:
- Resolution is the main cost driver (576×208 ≈ 2.4× the pixels of 224²); it is paid for because defects are small. If latency is too high, test 448×160 and compare recall.
- EfficientNet-B0 vs B1/ResNet50: larger models cost latency/size for gains that a ~250-positive dataset is unlikely to reveal.
- CPU vs GPU: GPU wins for batch/throughput; a single line camera at a few images/second is fine on CPU, which is cheaper to deploy. Measure before deciding.
- ONNX/quantisation: further speed-up options, deliberately left out of scope.
- **Why classification:** the requirement is a pass/fail decision. **Why not YOLO:** needs bounding-box labels and gives localisation that was not requested. **Why not PatchCore/anomaly detection:** it shines with few/no defect examples, unseen defect types, localisation needs, or highly variable defects; here we have hundreds of labelled defects, so supervised classification is simpler and normally more accurate. If new defect types dominate in production, add an anomaly model as a second opinion.

## 21. Known Limitations
- Not the client's dataset; one factory, one product family.
- Few positives overall (356): metrics have noticeable uncertainty; no repeated-seed/cross-validation study was done.
- Possible near-duplicate leakage in KSDD2 (no item IDs).
- No localisation; Grad-CAM is coarse (last conv stage) and is a debugging aid, not proof.
- Uncalibrated probabilities; threshold tuned on a small validation set.
- Non-commercial dataset licence.
- The Docker build/run and the live API demo are verified manually; the pipeline does not record them.

## 22. Production Considerations
Model versioning (tag image + checkpoint with git SHA, seed, threshold); monitoring (latency, error rates, defect-rate and confidence histograms → drift alarms); data drift (input brightness/contrast stats vs training); model drift (periodic labelled audit sample from human inspectors); retraining (add reviewed FNs/FPs, re-tune threshold); confidence thresholds per product line; human review for `review_recommended`; rollback by re-deploying the previous image tag. None of this infrastructure is implemented — intentionally, given scope.

## 23. Future Improvements
Cross-validation for confidence intervals; calibration (temperature scaling); tiling/higher resolution for tiny defects; add an anomaly-detection second opinion for unseen defects; ONNX/INT8 export; test-time augmentation if recall is short; active-learning loop from human-review decisions.

## 24. Conclusion
The repository implements the full pipeline the assessment asks for with a deliberately small, explainable footprint. Its credibility rests on the reported numbers, so before submission: run `./run_all.sh` (numbers are filled automatically from the generated files), write Sections 14–16 from the actual FP/FN images, verify Docker, and be ready to explain each choice above.
