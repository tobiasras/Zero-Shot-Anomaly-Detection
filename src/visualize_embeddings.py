import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from models.load_vision_transformer import get_vit_model
from util.logger import log
from util.paths import PROJECT_ROOT
from torchvision import transforms
from data.data_loader import DatasetLoader
import torch
import os

from util.visualization import embedding_to_rgb


def reverse_transform(tensor_img):
    """
    Reverses a PyTorch image transform: Normalize and ToTensor.

    Args:
        tensor_img: torch.Tensor of shape (C,H,W), normalized with ImageNet mean/std.

    Returns:
        np.ndarray: H x W x C image in uint8 [0-255] range, suitable for plt.imshow
    """
    # ImageNet mean/std
    mean = torch.tensor([0.485, 0.456, 0.406], device=tensor_img.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=tensor_img.device).view(3, 1, 1)

    # Undo normalization
    img = tensor_img * std + mean

    # Convert to numpy, HWC, and 0-255 uint8
    img = img.permute(1, 2, 0).detach().cpu().numpy()
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)

    return img


if __name__ == '__main__':
    log.info('Starting Zero-Shot Anomaly')
    matplotlib.use("TkAgg")  # or "Qt5Agg" if you have PyQt5 installed

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # convert grayscale to 3 channels
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    model = get_vit_model("dino_vits16").eval()

    data_type = "screw"
    path_to_data = PROJECT_ROOT / "data" / "mvtec_anomaly_detection" / data_type / "train" / "good"
    dataset = DatasetLoader(str(path_to_data), transform=transform)

    patch_size = 16

    # save_dir = PROJECT_ROOT / "embeddings" / data_type / "train" / "good"
    # os.makedirs(save_dir, exist_ok=True)

    for idx, data in enumerate(dataset):
        x = data.unsqueeze(0)
        patch_embeddings = model.get_intermediate_layers(x, n=1)[0][:, 1:, :]
        embedding = patch_embeddings[0].detach().cpu().numpy()

        rgb_values = embedding_to_rgb(embedding)
        print(rgb_values.shape)

        # org image:
        org_img = reverse_transform(data)

        print("img_shape")
        print()

        plt.imshow(org_img)  # convert to 0–255 image if not already
        plt.axis("off")  # hides axes
        plt.show()

        # embeddings visualized
        h, w = org_img.shape[0] // patch_size, org_img.shape[1] // patch_size  #
        img_restored = rgb_values.reshape(h, w, 3)  # for RGB
        original_img = data.permute(1, 2, 0).detach().cpu().numpy()  # (H, W, C)
        plt.imshow(img_restored)  # convert to 0–255 image if not already
        plt.axis("off")  # hides axes
        plt.show()

        break
