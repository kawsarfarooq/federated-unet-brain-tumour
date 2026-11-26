# Experiment records

## Completed experiment archive

All five training configurations completed, and each selected checkpoint passed reload verification.

| Configuration | Test Dice |
|---|---:|
| Centralized, width 16 | 0.8317 |
| Centralized, width 32 | 0.8099 |
| FedAvg, 2 clients | 0.7557 |
| FedAvg, 4 clients | 0.7028 |
| FedAvg, 8 clients | 0.5227 |

The archive preserves the original sampling and evaluation behavior described in reference 4 of the main README. The global federated validation selection contained only background slices; see the saved sampler audit. Federated scores use the archived joint evaluation. Full checkpoints and logs are retained in the local experiment outputs.

## Integration and software checks

Eleven automated tests passed, including CPU/GPU device selection. Centralized and two-client CPU integration runs completed two training steps each, with shared initialization and fresh-model checkpoint reload verification. Validation included 12 foreground slices from 32 selected slices.

See smoke/results.json and smoke/learning_curves.png for integration records. The complete-slice pipeline changes coverage, loss, and metric definitions. Integration checks do not constitute full benchmark measurements for that pipeline; the completed archive above belongs to the separately documented protocol.

See [protocol details](../docs/PROTOCOL.md#result-provenance) and [README references](../README.md#references).
