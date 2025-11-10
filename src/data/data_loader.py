from torchvision import transforms
from torch.utils.data import Dataset
import os
from PIL import Image


class DatasetLoader(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.images = list(root_dir.rglob("*.png"))
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path = self.images[idx].absolute()

        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, path
