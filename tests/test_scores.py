import numpy as np
from brainseg_fl.scores import SegmentationScores


def test_background_accuracy_cannot_hide_a_missed_tumour():
    scorer = SegmentationScores()
    empty = np.zeros((8, 2, 2, 1), dtype=np.uint8)
    positive = np.zeros((1, 2, 2, 1), dtype=np.uint8); positive[0, 0, 0, 0] = 1
    scorer.add("background_case", np.zeros_like(empty), empty)
    scorer.add("tumour_case", np.zeros_like(positive), positive)
    values = scorer.report()["metrics"]
    assert values["dice_all_slices_mean"] > 0.8
    assert values["dice_foreground_slices_mean"] == 0
    assert values["dice_foreground_patients_mean"] == 0
    assert values["dice_micro"] == 0
    assert values["foreground_slices"] == 1 and values["slices"] == 9


def test_perfect_predictions_have_unit_dice_and_zero_false_positives():
    y = np.zeros((2, 3, 3, 1), dtype=np.uint8); y[0, 1, 1, 0] = 1
    scorer = SegmentationScores(); scorer.add("case", y.astype(np.float32), y)
    values = scorer.report()["metrics"]
    for name in ["dice_micro", "dice_patient_mean", "dice_foreground_slices_mean", "iou_micro"]:
        assert values[name] == 1
    assert values["false_positive_rate"] == 0


def test_undefined_foreground_metrics_are_null_not_perfect_scores():
    y = np.zeros((1, 2, 2, 1), dtype=np.uint8)
    scorer = SegmentationScores(); scorer.add("empty", np.ones_like(y), y)
    values = scorer.report()["metrics"]
    assert values["dice_foreground_slices_mean"] is None
    assert values["dice_foreground_patients_mean"] is None
    assert values["false_positive_rate"] == 1
