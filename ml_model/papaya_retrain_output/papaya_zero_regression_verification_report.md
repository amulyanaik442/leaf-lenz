# Papaya Integration — Zero-Regression Verification Report

Date: 2026-08-05
Status: **PASS — ready for user approval before any production deploy**

## 1. Scope

Addition of **Papaya leaf disease detection** (5 classes) to the existing Leaf Lenz
general model using the approved modular workflow. Production assets
(`detector/ml_assets/model.onnx`, `class_names.json`) were **NOT modified**.
All verification was performed against the candidate
(`ml_model/papaya_retrain_output/papaya_candidate.onnx`, 146 classes).

## 2. Model

- Backbone: frozen production EfficientNet-B0 (1280-d), feature extraction from the
  deployed ONNX (no retraining of existing weights).
- New head: `Linear(1280 -> 5)`, papaya-only, appended as rows 141–145.
- New classes: papaya___anthracnose, papaya___bacterial_spot, papaya___curl,
  papaya___healthy, papaya___ring_spot.

## 3. Accuracy (internal Papaya_Split test set, 95 images, TTA 4-view)

| Metric | Value |
|---|---|
| Train accuracy | 99.59% |
| Best validation accuracy | 85.00% |
| Test accuracy (candidate) | **83.2% (79/95)** |

Per-class (test): anthracnose 13/19, bacterial_spot 18/19, curl 16/19,
healthy 17/19, ring_spot 15/19.

## 4. Zero Regression — weights

| Check | Result |
|---|---|
| Non-papaya weight max diff (rows 0–140) | 0.0 (byte-identical) |
| Non-papaya bias max diff | 0.0 (byte-identical) |
| Zero regression (weight level) | OK |

## 5. Zero Regression — inference

| Check | Samples | Result |
|---|---|---|
| Crop-filtered top-1 mismatch (candidate vs production), 9 existing crops | 447 | 0 |
| Max |logit diff| over first 141 classes | 447 | 0.0 |
| Larger empirical pass (earlier run) | 927 | 0 mismatches, diff 0.0 |
| Train-time inference spot check | 767 | 0 top-1 old-class changes |

Per-crop filtered mismatches: bean 0/52, cauliflower 0/18, pumpkin 0/60,
tea 0/60, cacao 0/60, chilli 0/48, turmeric 0/32, black_pepper 0/57, ginger 0/60.

## 6. Papaya false-positive audit on existing crops

Uncropped full-model argmax now selects a papaya class for 28/447 (6.3%) existing-crop
images; earlier larger pass 80/767 (10.4%). This is within the historical range for
previously deployed crops (bean 3.6%, turmeric 5.3%, black_pepper 9.0%, cacao 15.8%,
ginger 25.6%) and does **not** affect production predictions, because the app isolates
logits per crop via `CROP_CLASS_PREFIXES` prefix filtering (only the selected crop's
classes participate in softmax). The existing low-confidence mismatch gate
(`_predict_general_filtered`) also flags weak/forced predictions.

## 7. App wiring (added, non-destructive)

| Item | Status |
|---|---|
| `CROP_CLASS_PREFIXES['papaya'] = 'papaya___'` | Added |
| `index.html` dropdown Papaya option | Added |
| `disease_data.json` 5 papaya entries | Added |
| `predict_by_crop(..., 'papaya')` | Runs (falls back to full softmax until deploy — expected, no papaya rows in production yet) |

## 8. Deploy checklist (pending user approval)

1. Backup production `model.onnx` + `class_names.json`.
2. Copy `papaya_candidate.onnx` -> `detector/ml_assets/model.onnx`.
3. Copy `candidate_class_names.json` -> `detector/ml_assets/class_names.json`.
4. Restart the server (reloads the cached ONNX session) and re-run a live smoke test.
5. Optionally commit app wiring + candidate artifacts.

**No production change has been made. Awaiting approval.**
