import torch
from torchvision import transforms
from PIL import Image


class Transform:
    def __init__(self, resize, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.transform = transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=mean,
                std=std
            )])
    def get_transform(self):
        return self.transform

    def reverse_transform(self, tensor, original_size=None):

        # Denormalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        tensor = tensor * std + mean  # reverse normalization

        # Clamp to [0,1]
        tensor = tensor.clamp(0, 1)

        # Convert to PIL Image
        img = transforms.ToPILImage()(tensor)

        # Resize back if original_size is provided
        if original_size is not None:
            img = img.resize(original_size, Image.BILINEAR)

        return img
