"""Regenerate methodology and measured-result figures; optionally show dataset labels."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
COLORS = {"blue": "#dcecf9", "green": "#dff2e6", "orange": "#fff0da", "purple": "#ece5f7"}


def badges():
    """Create local SVG badges with font-independent text paths and vector icons."""
    from matplotlib.patches import Rectangle, Polygon
    plt.rcParams["svg.fonttype"] = "path"
    plt.rcParams["font.family"] = "DejaVu Sans"
    folder=FIGURES/"badges"
    folder.mkdir(exist_ok=True)
    for name,label,color,width in [
        ("python","Python 3.12","#356a95",1.4),
        ("tensorflow","TensorFlow 2.16.2","#b95a12",1.9),
        ("hardware","CPU / NVIDIA GPU","#5f4b8b",2.0),
        ("tests","Automated tests","#27734c",1.8),
    ]:
        fig,ax=plt.subplots(figsize=(width,.30))
        fig.subplots_adjust(0,0,1,1)
        fig.patch.set_facecolor(color)
        ax.set_xlim(0,width);ax.set_ylim(0,.3);ax.axis("off")
        if name == "tests":
            ax.plot([.08,.12,.21],[.15,.10,.21],color="white",lw=1.8)
        else:
            ax.add_patch(Rectangle((.09,.095),.12,.12,fill=False,edgecolor="white",lw=1))
            for x in [.12,.18]:
                ax.plot([x,x],[.07,.095],color="white",lw=1)
                ax.plot([x,x],[.215,.24],color="white",lw=1)
        ax.text(.29,.15,label,ha="left",va="center",color="white",fontsize=8.5,weight="normal")
        fig.savefig(folder/f"{name}.svg",facecolor=color,metadata={"Date":None})
        plt.close(fig)


def methodology():
    fig, ax = plt.subplots(figsize=(15, 9))
    ax.set_xlim(0, 15); ax.set_ylim(0, 9); ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.text(7.5, 8.7, "Federated U-Net: methodology", ha="center", fontsize=24, weight="bold", color="#172b4d")

    def box(x, y, w, h, text, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.12", facecolor=COLORS[color], edgecolor="#53657c", linewidth=1.2))
        ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=11, color="#172b4d", linespacing=1.5)

    def arrow(a, b, text=None):
        ax.annotate("", xy=b, xytext=a, arrowprops={"arrowstyle": "-|>", "lw": 1.5, "color": "#42566e"})
        if text:
            ax.text((a[0]+b[0])/2, (a[1]+b[1])/2+0.10, text, fontsize=9, ha="center", va="bottom", bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})

    box(3.9, 7.5, 7.2, .65, "MSD BrainTumour: 484 labeled patients", "blue")
    box(3.9, 6.35, 7.2, .7, "Patient-level split: 290 train / 97 validation / 97 test", "blue")
    box(3.9, 4.95, 7.2, .95, "Preprocess each split independently\nFLAIR / volume Z-score / 128 x 128 crop / axial slices / binary masks", "blue")
    arrow((7.5,7.5),(7.5,7.05)); arrow((7.5,6.35),(7.5,5.9))
    box(.5, 2.95, 5.3, 1.25, "CENTRALIZED TRAINING\nPool training patients / augment\nU-Net base widths 16 and 32", "green")
    box(6.5, 2.95, 5.3, 1.25, "SIMULATED FEDERATED TRAINING\nPartition training patients into 2, 4, or 8 clients\nGlobal width-16 U-Net -> one local epoch -> FedAvg", "orange")
    arrow((5.2,4.95),(3.15,4.2),"Training only")
    arrow((9,4.95),(9.15,4.2),"Training only")
    ax.annotate("",xy=(11.8,3.9),xytext=(11.8,3.15),arrowprops={"arrowstyle":"-|>","lw":1.5,"color":"#c87824","connectionstyle":"arc3,rad=1.6"})
    ax.text(13.0,3.5,"Repeat\nrounds",fontsize=9,ha="center",va="center",color="#87501b")
    box(3.9, 1.55, 7.2, .75, "Best checkpoint: validation foreground-patient mean Dice", "purple")
    arrow((3.15,2.95),(5.5,2.3)); arrow((9.15,2.95),(9,2.3))
    ax.plot([3.8,.15,.15,3.8],[5.3,5.3,1.95,1.95],color="#42566e",lw=1.3)
    arrow((3.45,1.95),(3.9,1.95))
    ax.text(.25,4.6,"Validation only",fontsize=9,ha="left",va="center",color="#42566e")
    box(3.9, .2, 7.2, .85, "Held-out test evaluation\nPatient Dice / foreground-slice Dice / IoU / precision / recall / FPR", "purple")
    arrow((7.5,1.55),(7.5,1.05))
    ax.plot([11.2,14.5,14.5,11.6],[5.3,5.3,.65,.65],color="#42566e",lw=1.3)
    arrow((11.6,.65),(11.1,.65))
    ax.text(14.35,4.6,"Test only",fontsize=9,ha="right",va="center",color="#42566e")
    fig.tight_layout()
    for ext in ["png", "svg"]:
        fig.savefig(FIGURES/f"methodology.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def baseline_comparison():
    records = json.loads((ROOT/"results/reproduction/comparison.json").read_text())
    names = ["Central\nwidth 16", "Central\nwidth 32", "FedAvg\n2 clients", "FedAvg\n4 clients", "FedAvg\n8 clients"]
    ours = [r["ours"]["dice_coef"] for r in records.values()]
    reported = [r["paper"]["dice_coef"] if r["paper"] else np.nan for r in records.values()]
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(names))
    a = ax.bar(x-.18, ours, width=.36, color="#3676b5", label="Reproduced baseline")
    b = ax.bar(x+.18, reported, width=.36, color="#c48a43", label="Published study")
    for bars in [a,b]:
        for bar in bars:
            h=bar.get_height()
            if np.isfinite(h): ax.text(bar.get_x()+bar.get_width()/2,h+.015,f"{h:.3f}",ha="center",fontsize=9)
    ax.set_xticks(x,names); ax.set_ylim(0,1); ax.set_ylabel("Test Dice")
    ax.set_title("Completed baseline reproduction",weight="bold",pad=16)
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(axis="y",alpha=.2); ax.set_axisbelow(True)
    ax.legend(loc="upper right",frameon=False)
    fig.text(.5,.01,"Separate source-faithful protocol; these are not current-implementation benchmark scores. Width-32 published value unavailable.",ha="center",fontsize=8,color="#526070")
    fig.tight_layout(rect=(0,.04,1,1))
    for ext in ["png", "svg"]: fig.savefig(FIGURES/f"baseline_comparison.{ext}",dpi=200,bbox_inches="tight")
    plt.close(fig)


def dataset_example(dataset):
    import nibabel as nib
    # A training patient under the saved seed-42 patient split.
    patient = "BRATS_434"
    image = np.asarray(nib.load(str(dataset/"imagesTr"/f"{patient}.nii.gz")).dataobj[...,0],dtype=np.float32)
    mask = np.asarray(nib.load(str(dataset/"labelsTr"/f"{patient}.nii.gz")).dataobj) > 0
    z=int(mask.sum(axis=(0,1)).argmax())
    x,y=image.shape[:2]; a=(x-128)//2; b=(y-128)//2
    image=image[a:a+128,b:b+128,z]; mask=mask[a:a+128,b:b+128,z]
    fig, axes=plt.subplots(1,3,figsize=(10,3.8))
    for ax in axes: ax.axis("off")
    for ax in [axes[0],axes[2]]: ax.imshow(image.T,cmap="gray",origin="lower")
    axes[1].imshow(mask.T,cmap="gray",origin="lower",vmin=0,vmax=1)
    axes[2].imshow(np.ma.masked_where(~mask.T,mask.T),cmap="autumn",alpha=.45,origin="lower",vmin=0,vmax=1)
    for ax,title in zip(axes,["FLAIR MRI", "Binary ground truth", "Ground-truth overlay"]): ax.set_title(title,fontsize=12)
    fig.suptitle(f"Dataset illustration | {patient} | axial slice {z} | no model predictions",fontsize=11)
    fig.text(.5,.01,"Source: Medical Segmentation Decathlon Task01_BrainTumour / BraTS. Image panel: CC BY-SA 4.0.",ha="center",fontsize=8)
    fig.tight_layout(rect=(0,.04,1,.94))
    fig.savefig(FIGURES/"dataset_example.png",dpi=200,bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",type=Path)
    options=parser.parse_args()
    FIGURES.mkdir(exist_ok=True)
    badges(); methodology(); baseline_comparison()
    if options.dataset: dataset_example(options.dataset)
    for svg in FIGURES.rglob("*.svg"):
        svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")
    print(f"Figures saved to {FIGURES}")
