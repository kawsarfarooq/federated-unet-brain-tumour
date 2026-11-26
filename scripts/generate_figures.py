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


def classic_badges():
    """Classic split-label badges, bundled logos, and text embedded as vector paths."""
    import xml.etree.ElementTree as ET
    from matplotlib.patches import Rectangle
    plt.rcParams["svg.fonttype"] = "path"
    plt.rcParams["font.family"] = "DejaVu Sans"
    folder=FIGURES/"badges"
    folder.mkdir(exist_ok=True)
    for name,label,value,color,left,right in [
        ("python","Python","3.12","#3776ab",70,40),
        ("tensorflow","TensorFlow","2.16.2","#ff6f00",92,52),
        ("task","Task","Binary Segmentation","#6750a4",38,125),
    ]:
        width=left+right
        fig,ax=plt.subplots(figsize=(width/96,20/96))
        fig.subplots_adjust(0,0,1,1)
        ax.set_xlim(0,width);ax.set_ylim(0,20);ax.axis("off")
        fig.patch.set_facecolor(color)
        ax.add_patch(Rectangle((0,0),left,20,color="#555555",linewidth=0))
        has_icon=name in ["python","tensorflow"]
        ax.text((24+left)/2 if has_icon else left/2,10,label,ha="center",va="center",fontsize=8.25,color="white")
        ax.text(left+right/2,10,value,ha="center",va="center",fontsize=8.25,color="white")
        target=folder/f"{name}.svg"
        fig.savefig(target,facecolor=color,metadata={"Date":None})
        plt.close(fig)
        if has_icon:
            svg=ET.parse(target)
            icon=ET.parse(FIGURES/"icons"/f"{name}.svg")
            ns="http://www.w3.org/2000/svg"
            group=ET.Element(f"{{{ns}}}g",{"transform":"translate(3, 2.25) scale(0.4375)"})
            for path in icon.getroot().iter(f"{{{ns}}}path"):
                group.append(ET.Element(f"{{{ns}}}path",{"d":path.attrib["d"],"fill":"white"}))
            svg.getroot().append(group)
            ET.register_namespace("",ns)
            svg.write(target,encoding="utf-8",xml_declaration=True)


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


def experiment_results():
    records = json.loads((ROOT/"results/reproduction/comparison.json").read_text())
    names = ["Central\nwidth 16", "Central\nwidth 32", "FedAvg\n2 clients", "FedAvg\n4 clients", "FedAvg\n8 clients"]
    scores = [record["ours"]["dice_coef"] for record in records.values()]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    bars = ax.bar(np.arange(len(names)), scores, width=.6,
                  color=["#3676b5", "#3676b5", "#c48a43", "#c48a43", "#c48a43"])
    for bar in bars:
        height=bar.get_height()
        ax.text(bar.get_x()+bar.get_width()/2,height+.015,f"{height:.4f}",ha="center",fontsize=10)
    ax.set_xticks(np.arange(len(names)),names)
    ax.set_ylim(0,1); ax.set_ylabel("Test Dice")
    ax.set_title("Segmentation results across training configurations",weight="bold",pad=15)
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(axis="y",alpha=.2); ax.set_axisbelow(True)
    fig.text(.5,.01,"Measured results from the completed experimental archive; protocol details are documented separately.",ha="center",fontsize=8,color="#526070")
    fig.tight_layout(rect=(0,.04,1,1))
    for ext in ["png", "svg"]:
        fig.savefig(FIGURES/f"experiment_results.{ext}",dpi=200,bbox_inches="tight")
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
    classic_badges(); methodology(); experiment_results()
    if options.dataset: dataset_example(options.dataset)
    for svg in FIGURES.rglob("*.svg"):
        svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")
    print(f"Figures saved to {FIGURES}")
