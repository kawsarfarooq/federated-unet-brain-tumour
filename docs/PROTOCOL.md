# Experimental protocol

## Data and training

Patient-disjoint seed-42 split: 290 training, 97 validation, and 97 test patients. Select FLAIR, standardize each volume, center-crop to 128 × 128, and merge nonzero mask labels. Full mode includes every axial slice. Training batches retain the final partial batch. Random flips and rotations apply only to training.

The U-Net has four encoder levels by default. Centralized widths are 16 and 32; federated width is 16 with 2, 4, or 8 clients. Client assignment is patient-level. Each client receives global weights, resets Adam state, and completes one local epoch per round. FedAvg weights updates by training-slice count. Configurations of the same architecture share initialization.

Adam uses learning rate 0.0001 and batch size 8. Loss is an equal-weight combination of binary cross-entropy and per-slice soft Dice loss. Default budgets are 10 centralized epochs and 10 rounds; early stopping patience is 5 checks. Budget options and stopping points must accompany reported scores.

## Evaluation

Validation foreground-patient mean Dice selects checkpoints. Test data is used after selection. Threshold probabilities at 0.5. Patient Dice pools voxel counts across a patient's selected slices before computing Dice. Micro Dice pools counts across patients. Foreground-slice Dice averages only slices containing positive target pixels; all-slice Dice includes empty slices. Empty/empty Dice is 1. Undefined foreground metrics are null. Precision and recall use zero for undefined denominators. Coverage counts and false positive rate accompany scores.

## Scope and limitations

The task is cropped 2-D binary segmentation, rather than full-resolution multiclass 3-D segmentation. Cropping may exclude peripheral anatomy. Clients are simulated in one process, without network communication, secure aggregation, or differential privacy. Patient partitions do not establish realistic institution-level distribution shifts. A single seed does not quantify variation across repeated training runs.

## Result provenance

results/reproduction contains completed baseline records from a separate source-faithful pipeline. results/smoke contains small execution checks of this implementation. Differences in sampling, loss, and metric definitions preclude treating these as interchangeable results. Full benchmark results for the present implementation have not been produced.
