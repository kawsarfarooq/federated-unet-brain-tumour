"""Sample-weighted FedAvg, following McMahan et al. (2017)."""
import numpy as np


def average_updates(updates):
    if not updates or any(count <= 0 for _, count in updates):
        raise ValueError("Every update needs a positive training-example count")
    template = updates[0][0]
    total = sum(count for _, count in updates)
    if any(len(weights) != len(template) for weights, _ in updates):
        raise ValueError("Incompatible model weight lists")
    combined = []
    for position, reference in enumerate(template):
        accumulator = np.zeros(reference.shape, dtype=np.float64)
        for weights, count in updates:
            candidate = weights[position]
            if candidate.shape != reference.shape or not np.isfinite(candidate).all():
                raise ValueError("Invalid or incompatible model parameters")
            accumulator += candidate.astype(np.float64) * (count / total)
        combined.append(accumulator.astype(reference.dtype))
    return combined
