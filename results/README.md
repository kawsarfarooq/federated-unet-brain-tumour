# Experiment records

## Completed source-faithful reproduction

All five experiments below completed using the separate reproduction pipeline. These are not results from brainseg_fl in this repository. The published numerical results were not matched across configurations.

| Configuration | Reproduced test Dice | Published test Dice |
|---|---:|---:|
| central_16 | 0.8317 | 0.8034 |
| central_32 | 0.8099 | not separately tabulated |
| fl_2 | 0.7557 | 0.6915 |
| fl_4 | 0.7028 | 0.7637 |
| fl_8 | 0.5227 | 0.7312 |

The original source sampler and evaluation behavior were preserved for the reproduction. The sampler audit found that the selected global federated validation slices were all background. Framework, random sequence, and execution differences limit exact numerical agreement. The federated values in comparison.json use the source-style joint evaluation. Full checkpoints, logs, and the source reproduction adapter are retained in the original local reproduction outputs; they are not bundled here.

## Integration and software checks

Eleven automated tests passed, including CPU/GPU device selection. A real-data CPU smoke run completed centralized and two-client training, each for two steps, with fresh-model checkpoint reload verification. Both configurations shared the same initialization. Validation included 12 foreground slices from 32 selected slices. These small runs verify execution and do not establish segmentation performance.

See smoke/results.json and smoke/learning_curves.png for the integration checks. The completed benchmark results above belong to the source-faithful reproduction protocol. The complete-slice pipeline changes coverage, loss, and metric definitions; the saved integration checks do not provide full benchmark measurements for that protocol.
