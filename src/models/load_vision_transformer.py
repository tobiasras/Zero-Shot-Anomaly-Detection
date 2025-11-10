from torchvision.models import VisionTransformer
import torch

def get_vit_model(model_name:str):
    if model_name == "dino_vits16":
        return torch.hub.load('facebookresearch/dino:main', 'dino_vits16')


    else:
        raise Exception(f"Unknown model name: {model_name}")







