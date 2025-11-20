from data.data_loader import DatasetLoader
from models.load_vision_transformer import get_vit_model
import random
import torch
from util.logger import log
import numpy as np
from sklearn.metrics import roc_auc_score
from util.paths import PROJECT_ROOT
from data.transform import Transform
from util.visualization import plot_sim_matrix, save_img, plot_roc_curve
from scipy.signal import correlate  # Import circular correlation function
from segment_anything import sam_model_registry, SamPredictor

def load_sam_model(checkpoint_path: str, model_type: str = "vit_h"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device)
    sam.eval()
    return sam, device

def sam_preprocess(model, x: torch.Tensor) -> torch.Tensor:
    """
    x: (B, 3, H, W) or (3, H, W), values in [0,1] from ToTensor.
    Only normalize with SAM's mean/std. **No resize.**
    """
    # Ensure batch dimension
    if x.dim() == 3:          # (3, H, W)
        x = x.unsqueeze(0)    # -> (1, 3, H, W)

    pixel_mean = model.pixel_mean.to(x.device)
    pixel_std = model.pixel_std.to(x.device)

    x = (x - pixel_mean) / pixel_std
    return x

def sam_encode_image(model, x: torch.Tensor) -> torch.Tensor:
    """
    Encode an image batch with SAM's image encoder and return patch embeddings.

    Input:
        x: (B, 3, H, W) or (3, H, W)
    Output:
        embeddings: (B, num_patches, C)
    """
    x = sam_preprocess(model, x)      # normalize only
    feats = model.image_encoder(x)    # (B, C, H', W')
    B, C, H, W = feats.shape
    feats = feats.view(B, C, H * W).permute(0, 2, 1)  # (B, num_patches, C)
    return feats

if __name__ == '__main__':
    log.info('Starting Zero-Shot Anomaly')

    seed = 42
    log.info('Setting seed: ' + str(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    object_type = "grid"
    log.info('Object Type : ' + str(object_type))
    test_path = PROJECT_ROOT / 'data' / 'mvtec_anomaly_detection' / object_type / 'test'
    ref_path = PROJECT_ROOT / 'data' / 'mvtec_anomaly_detection' / object_type / 'train'

    # Used for transforming images to dinoV3
    image_transformer = Transform(256)
    test_dataset = DatasetLoader(test_path, image_transformer.get_transform())
    ref_dataset = DatasetLoader(ref_path, image_transformer.get_transform())

    ref_count = 50
    indices = random.sample(range(len(ref_dataset)), ref_count)
    ref_images_tuple = [ref_dataset[i] for i in indices]
    ref_images, _ = zip(*ref_images_tuple)

    log.info('Total ref images: ' + str(len(ref_images)))

    ref_images_stack = torch.stack(ref_images)

    index = 0

    model_name = "SAM_vit_h_4b8939"
    log.info('Using model: ' + model_name)

    checkpoint_path = PROJECT_ROOT / 'models' / 'sam_vit_h_4b8939.pth'
    sam_model, device = load_sam_model(str(checkpoint_path), model_type="vit_h")

    ref_images_stack = ref_images_stack.to(device)

    with torch.no_grad():
        # -----------------------------------------------
        # Reference embeddings
        # -----------------------------------------------
        ref_embed = sam_encode_image(sam_model, ref_images_stack)  # (N_ref, P, C)
        ref_embed = ref_embed.mean(dim=0)  # Average across reference images -> (P, C)

        # Normalize reference embeddings
        ref_embed = torch.nn.functional.normalize(ref_embed, p=2, dim=-1)

        scores = []
        labels = []
        index = 0

        # -----------------------------------------------
        # Test loop
        # -----------------------------------------------
        for img, path in test_dataset:
            img = img.unsqueeze(0).to(device)  # (1,3,H,W)

            embed = sam_encode_image(sam_model, img)[0]  # (P_test, C)

            # Calculate circular correlation instead of direct distance
            sim_matrix = []
            test_vec = embed.cpu().numpy().flatten()  # 1D

            for ref_patch in ref_embed:  # Iterate over reference patches (C,)
                ref_vec = ref_patch.cpu().numpy().flatten()
                correlation = correlate(test_vec, ref_vec, mode='full')
                sim_matrix.append(correlation)

            sim_matrix = np.array(sim_matrix)

            # Percentile-based scoring (100th percentile == max)
            percentile = 100
            score = - np.percentile(sim_matrix.max(axis=1), percentile)

            scores.append((score, path))
            is_anomaly = "good" not in str(path)
            labels.append(1 if is_anomaly else 0)
            index += 1

    # ---------------------------------------------------
    # Visualization & metrics
    # ---------------------------------------------------
    # Average reference image for visualization
    ref_images_stack_cpu = ref_images_stack.cpu()
    avg_ref_img = ref_images_stack_cpu.mean(dim=0)  # (3,H,W)
    img_vis = image_transformer.reverse_transform(avg_ref_img)
    save_img(img_vis, PROJECT_ROOT / 'experiments' / 'figures', "ref_images_sam")

    values = np.array([s for s, _ in scores]).reshape(-1, 1)

    plot_roc_curve(labels, values)
    auc = roc_auc_score(labels, values)
    log.info(f"AUROC (SAM): {auc:.4f}")