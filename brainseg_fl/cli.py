"""Command-line entry point; device visibility is set before TensorFlow imports."""
import argparse
import os
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Federated U-Net course project")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--gpu", help="One physical GPU index or UUID")
    parser.add_argument("--gpu-memory", type=int, default=2560, help="TensorFlow memory cap in MiB")
    parser.add_argument("--clients", default=None, help="Comma-separated client counts; full default: 2,4,8")
    parser.add_argument("--width", type=int)
    parser.add_argument("--depth", type=int)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    options = parser.parse_args(argv)
    options.clients = [int(x) for x in (options.clients or ("2" if options.mode == "smoke" else "2,4,8")).split(",")]
    if options.width is None: options.width = 4 if options.mode == "smoke" else 16
    if options.depth is None: options.depth = 3 if options.mode == "smoke" else 4
    if options.epochs is None: options.epochs = 2 if options.mode == "smoke" else 10
    if options.rounds is None: options.rounds = 2 if options.mode == "smoke" else 10
    if min(options.clients + [options.width, options.depth, options.epochs, options.rounds,
                              options.batch_size, options.patience, options.threads, options.gpu_memory, options.image_size]) < 1:
        parser.error("Counts and sizes must be positive")
    if len(options.clients) != len(set(options.clients)) or options.learning_rate <= 0:
        parser.error("Client counts must be distinct and learning rate must be positive")
    if options.image_size % (2 ** options.depth):
        parser.error("Image size must be divisible by 2**depth")
    if options.device == "gpu":
        selected = options.gpu or os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if not selected or "," in selected or selected == "-1":
            parser.error("Choose exactly one GPU using --gpu or CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = selected
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["TF_USE_LEGACY_KERAS"] = "1"
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    from .training import execute
    execute(options)
    return 0
