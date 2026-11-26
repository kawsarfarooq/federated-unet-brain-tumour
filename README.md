<div align="center">

# Federated U-Net for Brain Tumour Segmentation

**Centralized and simulated federated learning for MRI segmentation**

**Author: [Kawsar Farooq (Kasa)](https://github.com/kawsarfarooq)**

![Python 3.12](figures/badges/python.svg)
![TensorFlow 2.16.2](figures/badges/tensorflow.svg)
![CPU and NVIDIA GPU](figures/badges/hardware.svg)
![Tests passing](figures/badges/tests.svg)

[Overview](#overview) Â· [Installation](#installation) Â· [Dataset](#dataset) Â· [Experiments](#run-experiments) Â· [Results](#experimental-results) Â· [References](REFERENCES.md)

</div>

## Overview

An implementation and evaluation framework for whole-tumour segmentation from FLAIR MRI. A 2-D U-Net is trained centrally or through example-weighted federated averaging (FedAvg), with patient-disjoint splits, explicit foreground metrics, and checkpoint verification.

![Methodology](figures/methodology.png)

*Methodology of the current implementation. Federated clients are simulated sequentially; validation selects checkpoints, and the test split is evaluated after selection.*

- **Centralized baselines:** U-Net base widths 16 and 32.
- **Federated configurations:** 2, 4, and 8 clients with a width-16 U-Net.
- **Auditable evaluation:** full-slice coverage, per-patient scores, foreground-slice metrics, and false positive rate.
- **Recorded experiments:** configuration, patient manifests, training histories, model weights, and reload checks.

## Folder structure

```text
.
â”œâ”€â”€ brainseg_fl/             # Data, U-Net, FedAvg, training, and metrics
â”œâ”€â”€ docs/                   # Protocol and experiment reporting guide
â”œâ”€â”€ figures/                # Methodology, dataset example, and result plots
â”œâ”€â”€ results/
â”‚   â”œâ”€â”€ reproduction/       # Completed baseline reproduction records
â”‚   â””â”€â”€ smoke/              # Pipeline integration checks
â”œâ”€â”€ scripts/                # Reproducible figure generation
â”œâ”€â”€ tests/                  # Data, aggregation, metrics, and checkpoint tests
â”œâ”€â”€ .github/workflows/      # Automated test workflow
â”œâ”€â”€ pyproject.toml
â”œâ”€â”€ requirements.txt
â”œâ”€â”€ REFERENCES.md
â””â”€â”€ ATTRIBUTION.md
```

## Installation

Use **Python 3.12**. The pinned environment uses TensorFlow 2.16.2 and legacy Keras 2.16.0.

```bash
git clone https://github.com/kawsarfarooq/federated-unet-brain-tumour.git
cd federated-unet-brain-tumour
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install pytest
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`. CPU execution is supported on Windows; NVIDIA GPU execution requires Linux with compatible drivers and TensorFlow CUDA dependencies. On Linux, install the matching CUDA dependencies with `python -m pip install 'tensorflow[and-cuda]==2.16.2'` when they are not already available.

## Dataset

Use [Medical Segmentation Decathlon Task01_BrainTumour](https://registry.opendata.aws/msd/). Extract the dataset outside the repository:

```text
/data/Task01_BrainTumour/
â”œâ”€â”€ imagesTr/     # 484 labeled MRI volumes
â”œâ”€â”€ labelsTr/     # Corresponding segmentation masks
â””â”€â”€ dataset.json
```

The default seed-42 split contains **290 training, 97 validation, and 97 test patients**. FLAIR is standardized per volume and center-cropped to 128 ÃN/A 128. Nonzero labels form the binary whole-tumour target; full mode processes every axial slice of the crop. Only training patients are assigned to clients.

![FLAIR and segmentation target](figures/dataset_example.png)

*Illustrative training-set slice and ground-truth binary mask. This is a dataset example, not a model prediction.*

Dataset terms and citations are listed in [REFERENCES.md](REFERENCES.md). Medical volumes are not included in this repository.

## Run experiments

### Quick verification on CPU

```bash
python -m brainseg_fl --dataset /data/Task01_BrainTumour --cache /data/brainseg_cache --run-dir runs/smoke --mode smoke --device cpu
```

This checks the pipeline using 4 training, 2 validation, and 2 test patients, 16 selected slices per patient, and two training steps. It verifies execution and saved-checkpoint inference; it is not a trained-model benchmark.

### Train and evaluate

```bash
python -m brainseg_fl --dataset /data/Task01_BrainTumour --cache /data/brainseg_cache --run-dir runs/full --mode full --device gpu --gpu 0
```

Use `--device gpu` for NVIDIA GPU execution or `--device cpu` for CPU execution. GPU 0 is selected by default; optionally pass `--gpu INDEX` or `--gpu GPU-UUID` to select another device. The TensorFlow GPU memory limit is configurable through `--gpu-memory`.

```bash
# CPU: the same complete experiment suite
python -m brainseg_fl --dataset /data/Task01_BrainTumour --cache /data/brainseg_cache --run-dir runs/full_cpu --mode full --device cpu

# GPU: use the default device
python -m brainseg_fl --dataset /data/Task01_BrainTumour --cache /data/brainseg_cache --run-dir runs/full_gpu --mode full --device gpu
```

| Setting | Default |
|---|---|
| Centralized epochs / federated rounds | 10 / 10 |
| Local epochs per client per round | 1 |
| Batch size | 8 |
| Optimizer / learning rate | Adam / 0.0001 |
| Loss | 0.5 binary cross-entropy + 0.5 soft Dice loss |
| Augmentation | Random flips and rotations |
| Checkpoint criterion | Validation foreground-patient mean Dice |
| Early stopping patience | 5 validation checks |

Set `--epochs 30 --rounds 200` for longer budgets. Report actual stopping points and processed examples when comparing configurations. See [the protocol](docs/PROTOCOL.md) for metric definitions and limitations.

### Output artifacts

Each run saves `configuration.json`, `data_manifest.json`, `results.json`, and `learning_curves.png`. Experiment directories contain histories and best/last weights. A fresh model reloads the selected checkpoint and verifies inference agreement.

## Experimental results

### Completed experiment suite

**All five reproduction experiments are complete:** two centralized models and three federated configurations. Every selected checkpoint was reloaded and verified. The following results use the source-faithful reproduction protocol; the complete-slice `brainseg_fl` pipeline has a different protocol, documented in [experimental scope](docs/PROTOCOL.md#result-provenance).

| Configuration | Reproduced test Dice | Reported test Dice |
|---|---:|---:|
| Centralized, width 16 | 0.8317 | 0.8034 |
| Centralized, width 32 | 0.8099 | â€” |
| FedAvg, 2 clients | 0.7557 | 0.6915 |
| FedAvg, 4 clients | 0.7028 | 0.7637 |
| FedAvg, 8 clients | 0.5227 | 0.7312 |

![Baseline reproduction comparison](figures/baseline_comparison.png)

Published values were not matched consistently. The validation sampler audit found all-background slices in the baseline federated validation selection. Framework and random-sequence differences also limit exact numerical agreement. See [experiment records](results/README.md), [comparison data](results/reproduction/comparison.json), and [sampler audit](results/reproduction/global_validation_sampler_audit.json).

### Learning curves

![Validation Dice histories](figures/reproduction_dice.png)

*Recorded validation Dice histories from the completed reproduction experiments. Centralized epochs and federated rounds have different update schedules; the federated validation sampler caveat is recorded in the audit.*

### Qualitative segmentation

![Segmentation examples from completed experiments](figures/reproduction_predictions.png)

*Held-out segmentation examples from the completed reproduction suite. These predictions correspond to the reproduction checkpoints, not the separate pipeline verification run. See [figure provenance](figures/README.md) for generation details.*

### Software validation

- **Automated tests:** data isolation, batch coverage, FedAvg weights, foreground metrics, model training, and checkpoint serialization.
- **Experiment integrity:** all five reproduction configurations completed with successful checkpoint reload checks.
- **Pipeline integration:** centralized and two-client CPU training completed on real data with shared initialization and verified checkpoint inference.

[Completion record](results/reproduction/completion.json) | [Detailed experiment records](results/README.md) | [Continuous integration](https://github.com/kawsarfarooq/federated-unet-brain-tumour/actions)

## Validation and figures

```bash
python -m pytest -q
python scripts/generate_figures.py
# Optional: regenerate the illustrative data panel locally
python scripts/generate_figures.py --dataset /data/Task01_BrainTumour
```

Tests cover patient isolation, complete batch coverage, aggregation weights, foreground metrics, trainability, and checkpoint reloads. The GitHub Actions workflow runs the same test suite on CPU.

## References and attribution

See [REFERENCES.md](REFERENCES.md) for U-Net, FedAvg, the dataset, and the motivating study, and [ATTRIBUTION.md](ATTRIBUTION.md) for source and license notes.
