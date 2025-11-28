from torchvision.models import VisionTransformer
import torch
from transformers import  AutoFeatureExtractor, AutoModel
from huggingface_hub import login
import os

def get_vit_model(model_name:str):
    if model_name == "dino_vits16":
        return torch.hub.load('facebookresearch/dino:main', 'dino_vits16')

    elif model_name == "dinov2_vits14":
        return torch.hub.load('facebookresearch/dinov2:main', 'dinov2_vits14')


    elif model_name == "dinov3_vits16":

        print(os.environ.get("HUGGINGFACE_TOKEN"))

        login(token=os.environ.get("HUGGINGFACE_TOKEN"))

        # Using Hugging Face DINOv3
        model_id = "facebook/dinov3-vits16-pretrain-lvd1689m"
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id)
        return {"model": model, "feature_extractor": feature_extractor}

    else:
        raise Exception(f"Unknown model name: {model_name}")







