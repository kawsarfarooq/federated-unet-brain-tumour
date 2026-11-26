import nibabel as nib
import numpy as np
import pytest
from brainseg_fl.data import PatientStore, SlicePool, split_patients, assign_clients


def test_every_slice_is_seen_once_including_partial_batch(tmp_path):
    root = tmp_path / "data"
    for name in ["imagesTr", "labelsTr"]: (root / name).mkdir(parents=True)
    for patient, count in [("a", 7), ("b", 9)]:
        image = np.zeros((24, 24, count, 4), dtype=np.float32)
        image[..., 0] = np.arange(count)[None, None, :]
        mask = np.zeros((24, 24, count), dtype=np.uint8); mask[10:14, 10:14, :] = 3
        nib.save(nib.Nifti1Image(image, np.eye(4)), root / "imagesTr" / f"{patient}.nii.gz")
        nib.save(nib.Nifti1Image(mask, np.eye(4)), root / "labelsTr" / f"{patient}.nii.gz")
    store = PatientStore(root, tmp_path / "cache", image_size=16)
    pool = SlicePool(store, ["a", "b"])
    seen = []; sizes = []
    for x, y, references in pool.batches(3, np.random.default_rng(9), augment=False):
        assert x.shape[1:] == y.shape[1:] == (16, 16, 1)
        assert np.isfinite(x).all() and set(np.unique(y)) == {0.0, 1.0}
        seen.extend(references); sizes.append(len(x))
    assert len(seen) == len(set(seen)) == 16
    assert set(seen) == {("a", i) for i in range(7)} | {("b", i) for i in range(9)}
    assert sizes[-1] == 1
    assert pool.coverage()["foreground_slices"] == 16


def test_splits_and_client_groups_never_share_a_patient():
    ids = [f"patient{i}" for i in range(20)]
    split = split_patients(ids, 42)
    assert set(split["train"]) | set(split["validation"]) | set(split["test"]) == set(ids)
    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        assert not set(split[a]) & set(split[b])
    assert split == split_patients(ids, 42)
    groups = assign_clients(split["train"], 8, 43)
    flat = [p for group in groups for p in group]
    assert len(flat) == len(set(flat)) and set(flat) == set(split["train"])
    with pytest.raises(ValueError): assign_clients(split["train"], 99, 43)
