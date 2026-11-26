"""Evaluation that keeps background-only slices visible rather than dominant."""
from collections import defaultdict
import numpy as np


def ratio(numerator, denominator, empty=1.0):
    return float(numerator / denominator) if denominator else empty


class SegmentationScores:
    def __init__(self):
        self.cases = defaultdict(lambda: dict(tp=0, fp=0, fn=0, tn=0, probability=0.0, truth=0, intersection=0.0))
        self.slice_dice = []
        self.foreground_dice = []
        self.loss_sum = 0.0

    def add(self, patient, probabilities, masks):
        masks = np.asarray(masks) > 0
        probabilities = np.asarray(probabilities, dtype=np.float64)
        if masks.shape != probabilities.shape or probabilities.ndim != 4:
            raise ValueError("Expected matching [slices, height, width, 1] arrays")
        if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
            raise ValueError("Predictions must be finite probabilities")
        prediction = probabilities >= 0.5
        case = self.cases[patient]
        for y, p, hard in zip(masks, probabilities, prediction):
            tp = int(np.count_nonzero(y & hard)); fp = int(np.count_nonzero(~y & hard))
            fn = int(np.count_nonzero(y & ~hard)); tn = int(np.count_nonzero(~y & ~hard))
            dice = ratio(2 * tp, 2 * tp + fp + fn)
            self.slice_dice.append(dice)
            if y.any():
                self.foreground_dice.append(dice)
            for key, value in dict(tp=tp, fp=fp, fn=fn, tn=tn).items():
                case[key] += value
            case["probability"] += float(p.sum()); case["truth"] += int(y.sum())
            case["intersection"] += float((p * y).sum())
            bounded = np.clip(p, 1e-7, 1 - 1e-7)
            bce = -np.mean(y * np.log(bounded) + (~y) * np.log1p(-bounded))
            soft = (2 * np.sum(p * y) + 1.0) / (np.sum(p) + np.sum(y) + 1.0)
            self.loss_sum += float(0.5 * bce + 0.5 * (1 - soft))

    def report(self):
        if not self.slice_dice:
            raise ValueError("Cannot evaluate an empty dataset")
        patients = []
        for patient, values in sorted(self.cases.items()):
            tp, fp, fn = (values[k] for k in ["tp", "fp", "fn"])
            patients.append({"patient": patient, **values, "dice": ratio(2 * tp, 2 * tp + fp + fn),
                             "soft_dice": ratio(2 * values["intersection"] + 1, values["probability"] + values["truth"] + 1),
                             "foreground_present": values["truth"] > 0})
        foreground_patients = [r["dice"] for r in patients if r["foreground_present"]]
        tp, fp, fn, tn = (sum(p[k] for p in patients) for k in ["tp", "fp", "fn", "tn"])
        return {"metrics": {
            "loss": self.loss_sum / len(self.slice_dice),
            "dice_micro": ratio(2 * tp, 2 * tp + fp + fn),
            "dice_patient_mean": float(np.mean([p["dice"] for p in patients])),
            "dice_foreground_patients_mean": float(np.mean(foreground_patients)) if foreground_patients else None,
            "dice_foreground_slices_mean": float(np.mean(self.foreground_dice)) if self.foreground_dice else None,
            "dice_all_slices_mean": float(np.mean(self.slice_dice)),
            "iou_micro": ratio(tp, tp + fp + fn), "precision_micro": ratio(tp, tp + fp, empty=0.0),
            "recall_micro": ratio(tp, tp + fn, empty=0.0), "false_positive_rate": ratio(fp, fp + tn, empty=0.0),
            "patients": len(patients), "slices": len(self.slice_dice),
            "foreground_patients": len(foreground_patients), "foreground_slices": len(self.foreground_dice)},
                "per_patient": patients}
