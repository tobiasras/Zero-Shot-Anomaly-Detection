import torch
import torch.nn.functional as F

# partly generated with chatgpt:
def calculate_distance(ref_patches: torch.Tensor,
                       image_patches: torch.Tensor,
                       measure_type: str = "cosine") -> torch.Tensor:

    measure_type = measure_type.lower()

    if measure_type == "cosine":
        ref_patches = ref_patches.mean(dim=0)
        return 1 - F.cosine_similarity(ref_patches, image_patches, dim=1)

    elif measure_type == "euclidean":
        ref_patches = ref_patches.mean(dim=0)
        # L2 distance
        return torch.norm(ref_patches - image_patches, p=2, dim=1)

    elif measure_type == "manhattan":
        ref_patches = ref_patches.mean(dim=0)
        # L1 distance
        return torch.norm(ref_patches - image_patches, p=1, dim=1)

    elif measure_type == "mahalanobis":
        N_ref, num_patches, D = ref_patches.shape
        ref_flat = ref_patches.reshape(-1, D)

        mean_ref = ref_flat.mean(dim=0)
        cov = torch.cov(ref_flat.T)  # (D, D)
        inv_cov = torch.linalg.pinv(cov)

        diff = image_patches - mean_ref  # shape (num_test_patches, D)
        dist = torch.sqrt(torch.sum(diff @ inv_cov * diff, dim=1))

        return dist
    else:
        raise ValueError(f"Unknown distance measure: {measure_type}")
