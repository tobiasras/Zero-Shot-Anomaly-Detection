import os

from sklearn.decomposition import PCA
import numpy as np
from .paths import PROJECT_ROOT
from PIL import Image
import torch
from pathlib import Path
import matplotlib

matplotlib.use('TkAgg')  # or 'Qt5Agg', 'Agg' for headless


def embedding_to_rgb(embedding):
    pca = PCA(n_components=3)
    emb = pca.fit_transform(embedding)

    emb_scaled = 255 * (emb - emb.min()) / (emb.max() - emb.min())
    emb_scaled = emb_scaled.astype(np.uint8)  # convert to uint8 if saving as image

    return emb_scaled


def plot_sim_matrix(sim_matrix, path, index):
    save_path = PROJECT_ROOT / "experiments" / "plots"
    save_path.mkdir(exist_ok=True, parents=True)

    # Determine title
    title = ""
    if "good" in str(path):
        title += "good"
    else:
        title += "anomaly"

    # Reshape 1D list into 2D image (assuming square)
    side = int(sim_matrix.numel() ** 0.5)
    sim_image = sim_matrix.view(side, side).numpy()

    # Plot as heatmap
    plt.figure(figsize=(6, 6))
    plt.suptitle(str(path), fontsize=10)  # add description under title

    plt.imshow(sim_image, cmap='viridis')  # choose colormap you like
    plt.colorbar(label='Similarity')
    plt.title("similary matrix" + title)
    plt.axis('off')  # hide axes

    # Save figure
    plt.savefig(str(save_path / f"{index} - {title}.png"), dpi=300, bbox_inches='tight')
    plt.close()


def save_img(img, path, title):
    path.mkdir(exist_ok=True, parents=True)

    plt.imshow(img)
    plt.savefig(str(path) + title, dpi=300, bbox_inches='tight')
    plt.close()


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def plot_roc_curve(y_true, y_scores, label=None):
    """
    Plots an ROC curve.

    Args:
        y_true (array-like): True binary labels (0 or 1).
        y_scores (array-like): Predicted scores or probabilities for the positive class.
        label (str, optional): Label for the curve in the legend.
    """

    path = PROJECT_ROOT / 'experiments' / 'figures' / "ref_images"
    path.mkdir(exist_ok=True, parents=True)

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'{label} (AUC = {roc_auc:.2f})' if label else f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(str(path) + "roc curve.png", dpi=300, bbox_inches='tight')


def visualize_anomaly(sim_matrix, path_to_img, topk, base_output, experiment_name, cmap="jet", alpha=0.5):
    # Convert to Path object
    p = Path(path_to_img)

    # Extract metadata
    error_type = p.parent.name            # "good" / "broken_large" / ...
    object_type = p.parents[2].name        # "bottle" / "cable" / ...

    # Build output path: outputs/experiment/object_type/error_type/
    save_dir = Path(base_output) / experiment_name / object_type / error_type
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save filename only
    filename = p.name
    save_path = save_dir / filename

    # Visualization Code
    img = Image.open(path_to_img).convert("RGB")
    img_np = np.array(img)

    sim_np = sim_matrix.detach().cpu().numpy()
    grid = int(np.sqrt(sim_np.shape[0]))
    sim_grid = sim_np.reshape(grid, grid)

    sim_up = Image.fromarray(sim_grid).resize((img_np.shape[1], img_np.shape[0]), Image.NEAREST)
    sim_up_np = np.array(sim_up)
    sim_norm = (sim_up_np - sim_up_np.min()) / (sim_up_np.max() - sim_up_np.min())

    plt.figure(figsize=(6, 6))
    plt.imshow(img_np)
    plt.imshow(sim_norm, cmap=cmap, alpha=alpha)
    plt.axis('off')

    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
