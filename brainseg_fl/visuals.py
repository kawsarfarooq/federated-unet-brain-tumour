"""Experiment plots with explicit foreground metrics and coverage."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def learning_curves(directory, plans):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, _, _ in plans:
        rows = json.loads((directory / name / "history.json").read_text())
        steps = [r["step"] for r in rows]
        axes[0].plot(steps, [r["validation"]["dice_foreground_patients_mean"] for r in rows], marker="o", label=name)
        axes[1].plot(steps, [r["validation"]["loss"] for r in rows], marker="o", label=name)
    axes[0].set(title="Validation: mean Dice over tumour-containing patients", ylabel="Patient Dice")
    axes[1].set(title="Validation: BCE + soft Dice loss", ylabel="Loss")
    for ax in axes:
        ax.set_xlabel("Centralized epoch / federated round")
        ax.grid(alpha=0.2); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(directory / "learning_curves.png", dpi=150)
    plt.close(fig)
