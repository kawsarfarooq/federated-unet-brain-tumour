# Federated U-Net for Brain Tumour Segmentation: An Implementation and Evaluation Study

A Deep Learning course project implementing centralized U-Net training and simulated federated averaging on Medical Segmentation Decathlon Task01_BrainTumour. This is a newly written implementation of established methods, with references in REFERENCES.md. It does not claim a novel architecture or federated algorithm.

## What this project contributes

- A modular implementation of patient splitting, NIfTI loading, U-Net, federated averaging, and evaluation.
- Complete traversal of the selected axial slices, including partial final batches, with disjoint patient-level train/validation/test sets.
- Explicit foreground coverage and separate patient, foreground-slice, and all-slice Dice metrics.
- A comparison of centralized training and 2-, 4-, and 8-client simulations with shared initialization and example-weighted aggregation.
- Saved manifests, histories, checkpoints, and checkpoint reload verification.

References and source acknowledgments are recorded in [REFERENCES.md](REFERENCES.md) and [ATTRIBUTION.md](ATTRIBUTION.md).

## Setup

Use Python 3.12 and a virtual environment:

```bash
python -m venv .venv
# Linux: source .venv/bin/activate
# Windows PowerShell: .venv/Scripts/Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -q
```

Download the public MSD Task01_BrainTumour dataset and extract it. Pass the directory containing imagesTr and labelsTr to --dataset. Keep the dataset outside this source package. Linux with a compatible NVIDIA TensorFlow installation is required for the GPU commands; Windows CPU runs are supported by the checked environment.

## Run a small verification

```bash
python -m brainseg_fl --dataset /data/Task01_BrainTumour --cache /data/brainseg_cache --run-dir runs/smoke --mode smoke --device cpu
```

This uses 4 training, 2 validation, and 2 test patients, 16 selected slices per patient, a small model, and two training steps. It checks execution and checkpoint integrity. Its low Dice scores are not evidence of trained-model performance. In the supplied workspace, run_smoke.ps1 uses the already downloaded dataset and Python environment.

## Full coursework comparison: one GPU at a time

```bash
python -m brainseg_fl --dataset /data/Task01_BrainTumour --cache /data/brainseg_cache --run-dir runs/full --mode full --device gpu --gpu 0
```

Choose one available GPU index or UUID in --gpu. Comma-separated selections are rejected. All configurations run sequentially on that single GPU. The default TensorFlow memory cap is 2560 MiB; adjust --gpu-memory if necessary.

Full mode runs centralized widths 16 and 32, then width-16 federated models with 2, 4, and 8 clients. Defaults are 10 centralized epochs and 10 federated rounds, with early stopping after 5 validation checks without improvement. Each client completes one local epoch per round. For a longer study use --epochs 30 --rounds 200, record this choice, and interpret comparisons using actual examples processed, time, and stopping points. Record the protocol and stopping criteria alongside reported scores.

## Protocol and interpretation

The seed-42 split assigns 290/97/97 of the 484 labeled patients to train/validation/test. Only training patients are partitioned between clients. The first modality (FLAIR) is standardized per volume, center-cropped to 128 by 128, and segmented into a binary whole-tumour mask (all nonzero labels). Full mode includes every axial slice of the crop. Cropping can remove peripheral anatomy; the task is cropped 2-D binary segmentation, not full-resolution multiclass 3-D segmentation.

Training uses Adam, learning rate 0.0001, batch size 8, random flips/rotations, and equal-weight binary cross-entropy plus soft Dice loss. Model width, depth, batch size, learning rate, seed, and client counts are configurable. Federated clients reset Adam state for each local update. Their weights are averaged by actual training-slice count. This simulates clients in one process; it does not demonstrate secure aggregation, differential privacy, hospital deployment, or network performance.

Checkpoints are selected using validation mean Dice over patients containing foreground. Evaluation thresholds probabilities at 0.5. Patient Dice pools slice counts per patient before averaging. Empty/empty Dice is 1; foreground-only metrics are null when undefined. Read these metrics together with coverage counts and false positive rate. Test labels are used only for final evaluation after checkpoint selection.

Outputs include configuration.json, data_manifest.json, results.json, learning_curves.png, per-experiment histories, and weights. A fresh model reloads each best checkpoint and verifies predictions. Verified real-data smoke results are included under results/smoke; full experiments with this implementation are pending. Separate experiment records are labeled by protocol under results/.

## Suggested report

Use REPORT_OUTLINE.md and fill it with your understanding, actual experiment settings, figures, and measured results. Use the citations in REFERENCES.md. Follow your course's policy on reused methods and AI assistance; disclose assistance accurately and ensure you can explain and modify the submitted code.

## Experiment records

See [results/README.md](results/README.md) for completed baseline experiment records and the verification results for this implementation. Results are labeled by protocol.
