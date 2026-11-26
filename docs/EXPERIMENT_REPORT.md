# Experiment reporting guide

Record the dataset provenance, patient split, seed, preprocessing, architecture, loss, optimizer, hardware, software versions, stopping criteria, and actual processed examples for each experiment.

Report patient Dice, foreground-slice Dice, false positive rate, elapsed time, coverage, and the selected epoch or round. Include validation learning curves and representative predictions with an explicit case-selection rule. Ground-truth examples must not be labeled as predictions. Separate baseline reproduction records from current-implementation benchmarks.

Discuss cropped 2-D inputs, binary targets, simulated clients, single-seed uncertainty, and absence of formal privacy guarantees. Cite the methods and data in REFERENCES.md.
