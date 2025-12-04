import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc
from imageio.v2 import imwrite
from .sam_segmentation import sam_segment_from_anomaly

from .paths import PROJECT_ROOT


# use any backend you like
matplotlib.use("TkAgg")  # or "Agg" on headless


def embedding_to_rgb(embedding):
    pca = PCA(n_components=3)
    emb = pca.fit_transform(embedding)

    emb_scaled = 255 * (emb - emb.min()) / (emb.max() - emb.min())
    emb_scaled = emb_scaled.astype(np.uint8)

    return emb_scaled


def plot_sim_matrix(sim_matrix, path, index):
    save_path = PROJECT_ROOT / "experiments" / "plots"
    save_path.mkdir(exist_ok=True, parents=True)

    title = "good" if "good" in str(path) else "anomaly"

    side = int(sim_matrix.numel() ** 0.5)
    sim_image = sim_matrix.view(side, side).numpy()

    plt.figure(figsize=(6, 6))
    plt.suptitle(str(path), fontsize=10)
    plt.imshow(sim_image, cmap="viridis")
    plt.colorbar(label="Similarity")
    plt.title("similarity matrix " + title)
    plt.axis("off")

    plt.savefig(save_path / f"{index} - {title}.png", dpi=300, bbox_inches="tight")
    plt.close()


def save_img(img, path, title):
    path.mkdir(exist_ok=True, parents=True)
    plt.imshow(img)
    plt.axis("off")
    plt.savefig(path / title, dpi=300, bbox_inches="tight")
    plt.close()


def plot_roc_curve(y_true, y_scores, label=None):
    path = PROJECT_ROOT / "experiments" / "figures" / "ref_images"
    path.mkdir(exist_ok=True, parents=True)

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 6))
    plt.plot(
        fpr,
        tpr,
        lw=2,
        label=f"{label} (AUC = {roc_auc:.2f})" if label else f"AUC = {roc_auc:.2f}",
    )
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC)")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(path / "roc_curve.png", dpi=300, bbox_inches="tight")
    plt.close()

def visualize_anomaly(sim_matrix, path_to_img, topk, base_output, experiment_name, cmap="jet", alpha=0.5):
    p = Path(path_to_img)
    error_type = p.parent.name       # 'good', 'broken_large'
    object_type = p.parents[2].name  # 'bottle', 'cable'

    img = Image.open(path_to_img).convert("RGB")
    img_np = np.array(img)
    H_img, W_img = img_np.shape[:2]

    sim_np = sim_matrix.detach().cpu().numpy()
    if sim_np.ndim == 1:
        grid = int(np.sqrt(sim_np.shape[0]))
        anomaly_grid = sim_np.reshape(grid, grid)
    else:
        anomaly_grid = sim_np  


    sam_mask = sam_segment_from_anomaly(img_np, anomaly_grid, num_points=5) 

    save_dir = Path(base_output) / object_type / error_type
    save_dir.mkdir(parents=True, exist_ok=True)
    vis_path = save_dir / p.name

    mask_dir = (
        PROJECT_ROOT
        / "experiments"
        / "masks_sam"
        / experiment_name
        / object_type
        / error_type
    )
    mask_dir.mkdir(parents=True, exist_ok=True)
    mask_path = mask_dir / p.name

    mask_rgb = np.zeros_like(img_np)
    mask_rgb[:, :, 0] = sam_mask * 255  

    mask_alpha = 0.45
    overlay = (img_np * (1 - mask_alpha) + mask_rgb * mask_alpha).astype(np.uint8)

    plt.figure(figsize=(6, 6))
    plt.imshow(overlay)
    plt.axis("off")
    plt.title(f"{object_type} - {error_type} - SAM")
    plt.savefig(vis_path, dpi=300, bbox_inches="tight")
    plt.close()

    imwrite(mask_path, (sam_mask * 255).astype("uint8"))

