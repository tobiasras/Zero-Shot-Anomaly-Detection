import torch
import torch.nn.functional as F


# partly generated with chatgpt:
def calculate_distance(ref_patches: torch.Tensor,
                       image_patches: torch.Tensor,
                       measure_type: str = "cosine", ref_aggregation_method='mean') -> torch.Tensor:
    measure_type = measure_type.lower()

    if measure_type == "cosine":
        ref_patch = ref_aggregation(ref_patches, image_patches, ref_aggregation_method)
        return F.cosine_similarity(ref_patch, image_patches, dim=1)  # higher = more similar

    elif measure_type == "euclidean":
        ref_patch = ref_aggregation(ref_patches, image_patches, ref_aggregation_method)
        return torch.norm(ref_patch - image_patches, p=2, dim=1)

    elif measure_type == "manhattan":
        ref_patch = ref_aggregation(ref_patches, image_patches, ref_aggregation_method)
        return torch.norm(ref_patch - image_patches, p=1, dim=1)

    elif measure_type == "mahalanobis":
        N_ref, num_patches, D = ref_patches.shape
        ref_flat = ref_patches.reshape(-1, D)

        mean_ref = ref_flat.mean(dim=0)
        cov = torch.cov(ref_flat.T)  # (D, D)
        inv_cov = torch.linalg.pinv(cov)

        diff = image_patches - mean_ref  # shape (num_test_patches, D)
        return torch.sqrt(torch.sum(diff @ inv_cov * diff, dim=1))

    else:
        raise ValueError(f"Unknown distance measure: {measure_type}")


def ref_aggregation(ref_patches, image_patches, method: str):
    if method == "mean":
        ref_patch = ref_patches.mean(dim=0)
    elif method == "median":
        ref_patch = ref_patches.median(dim=0).values
    elif method == "max":
        ref_patch = ref_patches.max(dim=0).values
    else:
        raise ValueError(f"Unknown ref_aggregation: {ref_aggregation}")

    return ref_patch
