"""Patient-separated data preparation and complete axial-slice traversal.

Only FLAIR is read. Cached arrays are cropped and normalized inputs, rather
than copies of all four MRI modalities. Dataset attribution: README.md#references.
"""
from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np


def discover_patients(root):
    root = Path(root)
    patients = sorted(p.name.removesuffix(".nii.gz") for p in (root / "labelsTr").glob("*.nii.gz") if not p.name.startswith("._"))
    if not patients:
        raise ValueError("No labelled .nii.gz files found under labelsTr")
    for patient in patients:
        if not (root / "imagesTr" / f"{patient}.nii.gz").is_file():
            raise FileNotFoundError(f"Missing image for {patient}")
    return patients


def split_patients(patients, seed):
    if len(patients) < 8 or len(patients) != len(set(patients)):
        raise ValueError("At least eight distinct labelled patients are required")
    ordered = np.random.default_rng(seed).permutation(sorted(patients)).tolist()
    stop = int(len(ordered) * 0.6)
    middle = stop + (len(ordered) - stop) // 2
    return {"train": ordered[:stop], "validation": ordered[stop:middle], "test": ordered[middle:]}


def assign_clients(patients, count, seed):
    if count < 1 or count > len(patients):
        raise ValueError("Client count must allow at least one patient per client")
    shuffled = np.random.default_rng(seed).permutation(patients)
    return [part.tolist() for part in np.array_split(shuffled, count)]


class PatientStore:
    def __init__(self, dataset, cache, image_size=128, slice_limit=0):
        self.dataset = Path(dataset).resolve()
        self.cache = Path(cache).resolve()
        self.image_size = image_size
        self.slice_limit = slice_limit
        self.details = {}
        self.open_arrays = OrderedDict()

    def prepare(self, patient):
        if patient in self.details:
            return self.details[patient]
        image_path = self.dataset / "imagesTr" / f"{patient}.nii.gz"
        mask_path = self.dataset / "labelsTr" / f"{patient}.nii.gz"
        signature = [str(image_path), image_path.stat().st_size, image_path.stat().st_mtime_ns,
                     mask_path.stat().st_size, mask_path.stat().st_mtime_ns,
                     self.image_size, self.slice_limit, "flair-normalize-crop-v1"]
        key = hashlib.sha256(json.dumps(signature).encode()).hexdigest()[:20]
        directory = self.cache / key
        marker = directory / "ready.json"
        if marker.exists():
            info = json.loads(marker.read_text())
        else:
            image_object = nib.load(str(image_path))
            if len(image_object.shape) != 4:
                raise ValueError(f"Expected a multi-modal 4-D image: {patient}")
            image = np.asarray(image_object.dataobj[..., 0], dtype=np.float32)
            mask = np.asarray(nib.load(str(mask_path)).dataobj)
            if image.shape != mask.shape or min(image.shape[:2]) < self.image_size:
                raise ValueError(f"Image/mask size mismatch or crop too large: {patient}")
            center = [(length - self.image_size) // 2 for length in image.shape[:2]]
            original_shape = list(image.shape)
            image = (image - float(image.mean())) / max(float(image.std()), 1e-6)
            region = (slice(center[0], center[0] + self.image_size), slice(center[1], center[1] + self.image_size))
            image = image[region]
            mask = mask[region] > 0
            selected = np.arange(image.shape[2])
            if self.slice_limit and self.slice_limit < len(selected):
                selected = np.linspace(0, len(selected) - 1, self.slice_limit, dtype=int)
            images = np.ascontiguousarray(np.moveaxis(image[:, :, selected], 2, 0)[..., None])
            masks = np.ascontiguousarray(np.moveaxis(mask[:, :, selected], 2, 0)[..., None].astype(np.uint8))
            info = {"patient": patient, "original_shape": original_shape, "axial_indices": selected.tolist(),
                    "slice_count": len(selected), "foreground_slices": int(masks.any(axis=(1, 2, 3)).sum()),
                    "crop": {"top": center[0], "left": center[1], "size": self.image_size}}
            directory.mkdir(parents=True, exist_ok=True)
            for name, array in [("images", images), ("masks", masks)]:
                temporary = directory / f"{name}.{os.getpid()}.tmp"
                with temporary.open("wb") as stream:
                    np.save(stream, array, allow_pickle=False)
                temporary.replace(directory / f"{name}.npy")
            temporary = directory / f"ready.{os.getpid()}.tmp"
            temporary.write_text(json.dumps(info, indent=2))
            temporary.replace(marker)
        self.details[patient] = {**info, "cache_directory": str(directory)}
        return self.details[patient]

    def arrays(self, patient):
        if patient not in self.open_arrays:
            info = self.prepare(patient)
            directory = Path(info["cache_directory"])
            self.open_arrays[patient] = tuple(np.load(directory / f"{name}.npy", mmap_mode="r", allow_pickle=False)
                                              for name in ["images", "masks"])
        self.open_arrays.move_to_end(patient)
        while len(self.open_arrays) > 8:
            self.open_arrays.popitem(last=False)
        return self.open_arrays[patient]


class SlicePool:
    def __init__(self, store, patients):
        self.store = store
        self.patients = list(patients)
        if len(self.patients) != len(set(self.patients)):
            raise ValueError("Duplicate patient in a pool")
        self.rows = [(patient, position) for patient in self.patients
                     for position in range(store.prepare(patient)["slice_count"])]

    def __len__(self):
        return len(self.rows)

    def coverage(self):
        return {"patients": len(self.patients), "slices": len(self),
                "foreground_slices": sum(self.store.prepare(p)["foreground_slices"] for p in self.patients)}

    def batches(self, batch_size, rng, augment=True):
        order = rng.permutation(len(self))
        for start in range(0, len(order), batch_size):
            references = [self.rows[int(index)] for index in order[start:start + batch_size]]
            images, masks = [], []
            for patient, position in references:
                x, y = self.store.arrays(patient)
                image, mask = x[position].copy(), y[position].astype(np.float32)
                if augment:
                    for axis in [0, 1]:
                        if rng.random() < 0.5:
                            image, mask = np.flip(image, axis), np.flip(mask, axis)
                    turns = int(rng.integers(0, 4))
                    image, mask = np.rot90(image, turns), np.rot90(mask, turns)
                images.append(image)
                masks.append(mask)
            yield np.stack(images), np.stack(masks), references
