"""Centralized and sequential federated experiments with complete coverage."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import time

import numpy as np
import tensorflow as tf

from . import TITLE
from .averaging import average_updates
from .data import PatientStore, SlicePool, discover_patients, split_patients, assign_clients
from .network import make_unet, segmentation_loss
from .scores import SegmentationScores


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def configure_runtime(options):
    tf.config.threading.set_intra_op_parallelism_threads(options.threads)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.experimental.enable_tensor_float_32_execution(False)
    visible = tf.config.list_physical_devices("GPU")
    if options.device == "gpu":
        if len(visible) != 1:
            raise RuntimeError("GPU mode requires exactly one visible GPU")
        tf.config.set_logical_device_configuration(visible[0], [tf.config.LogicalDeviceConfiguration(memory_limit=options.gpu_memory)])
    elif visible:
        raise RuntimeError("CPU mode must hide GPUs before importing TensorFlow")
    return [device.name for device in visible]


def new_model(options, width):
    tf.keras.backend.clear_session()
    random.seed(options.seed); np.random.seed(options.seed); tf.random.set_seed(options.seed)
    model = make_unet(options.image_size, width, options.depth, options.seed)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=options.learning_rate), loss=segmentation_loss)
    model.optimizer.build(model.trainable_variables)
    initial_optimizer = [v.numpy().copy() for v in model.optimizer.variables()]
    return model, initial_optimizer


def fit_one_pass(model, pool, batch_size, seed):
    rng = np.random.default_rng(seed)
    processed = 0; weighted_loss = 0.0
    for x, y, _ in pool.batches(batch_size, rng, augment=True):
        loss = float(model.train_on_batch(x, y, return_dict=True)["loss"])
        if not np.isfinite(loss):
            raise FloatingPointError("Training produced a non-finite loss")
        processed += len(x); weighted_loss += len(x) * loss
    if processed != len(pool):
        raise RuntimeError("An epoch did not traverse every training slice")
    return {"loss": weighted_loss / processed, "processed_examples": processed}


def evaluate(model, pool, batch_size):
    scores = SegmentationScores()
    for patient in pool.patients:
        x, y = pool.store.arrays(patient)
        for start in range(0, len(x), batch_size):
            probability = model(np.asarray(x[start:start + batch_size]), training=False).numpy()
            scores.add(patient, probability, y[start:start + batch_size])
    result = scores.report()
    if result["metrics"]["slices"] != len(pool):
        raise RuntimeError("Evaluation coverage is incomplete")
    return result


def run_experiment(options, name, width, pools, client_count=None):
    output = options.run_dir / name
    output.mkdir(parents=True, exist_ok=True)
    model, optimizer_initial = new_model(options, width)
    initial_weights = [a.copy() for a in model.get_weights()]
    best_score = -1.0; best_step = 0; stale = 0; rows = []
    started = time.time(); processed_total = 0
    train, validation, test = pools
    client_pools = []
    if client_count:
        groups = assign_clients(train.patients, client_count, options.seed + 1)
        client_pools = [SlicePool(train.store, group) for group in groups]
        save_json(output / "client_assignments.json", {str(i): p.patients for i, p in enumerate(client_pools)})
    budget = options.rounds if client_count else options.epochs
    for step in range(1, budget + 1):
        if client_count:
            global_weights = [a.copy() for a in model.get_weights()]
            updates = []; local_reports = []
            for client, pool in enumerate(client_pools):
                model.set_weights(global_weights)
                for variable, value in zip(model.optimizer.variables(), optimizer_initial):
                    variable.assign(value)
                report = fit_one_pass(model, pool, options.batch_size, options.seed + step * 1009 + client * 71)
                updates.append((model.get_weights(), len(pool)))
                local_reports.append({"client": client, **report})
            model.set_weights(average_updates(updates))
            seen = sum(r["processed_examples"] for r in local_reports)
            train_loss = sum(r["loss"] * r["processed_examples"] for r in local_reports) / seen
        else:
            report = fit_one_pass(model, train, options.batch_size, options.seed + step * 1009)
            seen = report["processed_examples"]; train_loss = report["loss"]; local_reports = []
        processed_total += seen
        validation_report = evaluate(model, validation, options.batch_size)
        selection = validation_report["metrics"]["dice_foreground_patients_mean"]
        if selection is None:
            raise ValueError("Validation contains no tumour foreground; checkpoint selection is undefined")
        row = {"step": step, "training_loss": train_loss, "processed_examples": seen,
               "cumulative_training_examples": processed_total, "validation": validation_report["metrics"], "clients": local_reports}
        rows.append(row)
        model.save_weights(str(output / "last.weights.h5"))
        if selection > best_score:
            best_score = selection; best_step = step; stale = 0
            model.save_weights(str(output / "best.weights.h5"))
        else:
            stale += 1
        save_json(output / "history.json", rows)
        print(f"{name}: {step}/{budget}; foreground-patient validation Dice={selection:.4f}", flush=True)
        if stale >= options.patience:
            break
    model.load_weights(str(output / "best.weights.h5"))
    test_report = evaluate(model, test, options.batch_size)
    result = {"title": TITLE, "scope": "real-data smoke verification" if options.mode == "smoke" else "course-project experiment",
              "full_dataset_evaluation": options.mode == "full", "experiment": name, "width": width,
              "clients": client_count or 0, "model_parameters": model.count_params(),
              "completed_steps": len(rows), "best_step": best_step, "training_examples_seen": processed_total,
              "training_slice_count": len(train), "validation_coverage": validation.coverage(),
              "test": test_report, "elapsed_seconds": time.time() - started,
              "initial_weights_sha256": hashlib.sha256(b"".join(a.tobytes() for a in initial_weights)).hexdigest()}
    # Reload into the same architecture and compare real-data inference.
    patient = test.patients[0]; images, _ = test.store.arrays(patient)
    probe = np.asarray(images[:2])
    expected = model(probe, training=False).numpy()
    reloaded = make_unet(options.image_size, width, options.depth, options.seed + 7919)
    reloaded.load_weights(str(output / "best.weights.h5"))
    np.testing.assert_allclose(expected, reloaded(probe, training=False).numpy(), rtol=1e-5, atol=1e-6)
    result["checkpoint_reload_verified"] = True
    save_json(output / "results.json", result)
    return result


def execute(options):
    options.run_dir.mkdir(parents=True, exist_ok=True)
    devices = configure_runtime(options)
    patients = discover_patients(options.dataset)
    split = split_patients(patients, options.seed)
    if options.mode == "smoke":
        split = {k: values[:(4 if k == "train" else 2)] for k, values in split.items()}
    store = PatientStore(options.dataset, options.cache, options.image_size, 16 if options.mode == "smoke" else 0)
    print("Preparing FLAIR slices and binary whole-tumour masks...", flush=True)
    pools = tuple(SlicePool(store, split[name]) for name in ["train", "validation", "test"])
    coverage = {name: pool.coverage() for name, pool in zip(split, pools)}
    if not coverage["validation"]["foreground_slices"]:
        raise ValueError("The selected validation subset has no tumour slices")
    save_json(options.run_dir / "data_manifest.json", {"split": split, "coverage": coverage, "patients": store.details})
    configuration = {k: str(v) if isinstance(v, Path) else v for k, v in vars(options).items()}
    save_json(options.run_dir / "configuration.json", {"title": TITLE, "options": configuration, "visible_gpus": devices,
              "tensorflow_float32_mode": "TF32 disabled", "versions": {p: importlib.metadata.version(p) for p in ["tensorflow", "tf-keras", "numpy", "nibabel"]}})
    print("Validation coverage:", coverage["validation"], flush=True)
    plans = [(f"central_w{options.width}", options.width, None)]
    if options.mode == "full":
        plans.append((f"central_w{options.width * 2}", options.width * 2, None))
    plans.extend((f"federated_{k}", options.width, k) for k in options.clients)
    results = []
    for name, width, clients in plans:
        results.append(run_experiment(options, name, width, pools, clients))
        save_json(options.run_dir / "results.json", results)
    from .visuals import learning_curves
    learning_curves(options.run_dir, plans)
    print("Course-project workflow complete:", options.run_dir, flush=True)
    return results
