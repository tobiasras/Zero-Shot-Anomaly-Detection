import torch
import torch.nn.functional as F

def calculate_distance(ref_patches: torch.Tensor, image_patches: torch.Tensor, measure_type=None) -> torch.Tensor:
    cos_sim = F.cosine_similarity(ref_patches, image_patches, dim=1)
    return cos_sim