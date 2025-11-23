from data.data_loader import DatasetLoader
import random
import torch
from util.logger import log
import numpy as np
from sklearn.metrics import roc_auc_score
from util.paths import PROJECT_ROOT
from data.transform import Transform
from util.visualization import save_img, plot_roc_curve
from segment_anything import sam_model_registry
from torchvision.transforms import functional as F

def load_sam_model(checkpoint_path: str, model_type: str = "vit_h"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device)
    sam.eval()
    return sam, device

def sam_preprocess(model, x: torch.Tensor) -> torch.Tensor:
    """
    x: (B, 3, H, W) or (3, H, W), values in [0,1].
    Resize to 1024x1024, normalize with SAM's mean/std.
    """
    if x.dim() == 3:
        x = x.unsqueeze(0)

    x = F.resize(x, size=(1024, 1024))
    
    pixel_mean = model.pixel_mean.to(x.device)
    pixel_std = model.pixel_std.to(x.device)

    x = (x - pixel_mean) / pixel_std
    return x

def sam_encode_image(model, x: torch.Tensor) -> torch.Tensor:
    """
    Encode a single image or batch with SAM's image encoder.
    x: (B, 3, H, W) or (3, H, W)
    Returns: (B, num_patches, C)
    """
    x = sam_preprocess(model, x)
    feats = model.image_encoder(x)
    B, C, H, W = feats.shape
    feats = feats.view(B, C, H * W).permute(0, 2, 1)  # (B, H*W, C)
    return feats

if __name__ == '__main__':
    log.info('Starting Zero-Shot Anomaly with SAM')

    seed = 42
    log.info('Setting seed: ' + str(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    object_type = "grid"
    log.info('Object Type : ' + str(object_type))
    test_path = PROJECT_ROOT / 'data' / 'mvtec_anomaly_detection' / object_type / 'test'
    ref_path = PROJECT_ROOT / 'data' / 'mvtec_anomaly_detection' / object_type / 'train'

    image_transformer = Transform(256)
    test_dataset = DatasetLoader(test_path, image_transformer.get_transform())
    ref_dataset = DatasetLoader(ref_path, image_transformer.get_transform())

    ref_count = 10  # REDUCED from 50 to save memory
    indices = random.sample(range(len(ref_dataset)), ref_count)
    
    model_name = "SAM_vit_h_4b8939"
    log.info('Using model: ' + model_name)

    checkpoint_path = PROJECT_ROOT / 'checkpoints' / 'sam_vit_h_4b8939.pth'
    log.info(f'Loading SAM from: {checkpoint_path}')
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"SAM checkpoint not found: {checkpoint_path}")
    
    sam_model, device = load_sam_model(str(checkpoint_path), model_type="vit_h")
    log.info(f'Using device: {device}')

    # Process reference images ONE AT A TIME to avoid memory crash
    print("[DEBUG] Encoding reference images (one at a time)...")
    ref_pool_list = []
    
    with torch.no_grad():
        for ref_idx, idx in enumerate(indices):
            img_tensor, _ = ref_dataset[idx]
            img_tensor = img_tensor.unsqueeze(0).to(device)  # (1, 3, H, W)
            
            # Encode single image
            ref_embed = sam_encode_image(sam_model, img_tensor)  # (1, num_patches, C)
            ref_embed = ref_embed[0]  # (num_patches, C)
            ref_embed = torch.nn.functional.normalize(ref_embed, p=2, dim=1)
            
            ref_pool_list.append(ref_embed.cpu())  # Move to CPU to free GPU memory
            
            if (ref_idx + 1) % 5 == 0:
                print(f"[PROGRESS] Encoded {ref_idx + 1}/{ref_count} reference images")
            
            del img_tensor, ref_embed
            torch.cuda.empty_cache()
        
        # Concatenate all reference patches
        ref_pool = torch.cat(ref_pool_list, dim=0).to(device)  # (N_ref_patches, C)
        print(f"[DEBUG] ref_pool shape: {ref_pool.shape}")

        scores = []
        labels = []
        test_len = len(test_dataset)

        # Test loop (also process one at a time)
        print("[DEBUG] Processing test images...")
        for idx, (img, path) in enumerate(test_dataset):
            img = img.unsqueeze(0).to(device)  # (1, 3, H, W)

            embed_batch = sam_encode_image(sam_model, img)  # (1, num_patches, C)
            embed = embed_batch[0]  # (num_patches, C)
            embed = torch.nn.functional.normalize(embed, p=2, dim=1)

            # Per-patch min distance to pooled reference patches
            dists = torch.cdist(embed, ref_pool)  # (P_test, N_ref_patches)
            per_patch_min = dists.min(dim=1).values  # (P_test,)

            # Use mean of min distances as anomaly score
            score = per_patch_min.mean().item()

            scores.append(score)
            is_anomaly = "good" not in str(path)
            labels.append(1 if is_anomaly else 0)

            # Progress output
            if (idx + 1) % 5 == 0 or (idx + 1) == test_len:
                print(f"[PROGRESS] {idx + 1}/{test_len} score={score:.4f} anomaly={is_anomaly}")

            del img, embed_batch, embed, dists, per_patch_min
            torch.cuda.empty_cache()

    # Visualization & metrics
    values = np.array(scores)
    plot_roc_curve(labels, values)
    auc = roc_auc_score(labels, values)
    log.info(f"AUROC (SAM): {auc:.4f}")
    print(f"\n=== AUROC (SAM): {auc:.4f} ===")