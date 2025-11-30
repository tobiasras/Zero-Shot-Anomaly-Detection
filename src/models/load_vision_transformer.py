from torchvision.models import VisionTransformer
import torch

from huggingface_hub import login
import os
import transformers
from util.logger import log


def get_vit_model(model_name:str):
    if model_name == "dino_vits16":
        return torch.hub.load('facebookresearch/dino:main', 'dino_vits16').eval()

    elif model_name == "dinov2_vits14":
        return torch.hub.load('facebookresearch/dinov2:main', 'dinov2_vits14').eval()


    #elif model_name == "dinov3_vits16":
#
    #    log.info(os.environ.get("HUGGINGFACE_TOKEN"))
    #    log.info(transformers.__version__)
#
    #    #login(token=os.environ.get("HUGGINGFACE_TOKEN"))
#
    #    # Using Hugging Face DINOv3
    #    model_id = "facebook/dinov3-vits16-pretrain-lvd1689m"
#
    #    processor = AutoImageProcessor.from_pretrained(model_id, token=os.environ.get("HUGGINGFACE_TOKEN"))
    #    model = AutoModel.from_pretrained(
    #        model_id,
    #        token=os.environ.get("HUGGINGFACE_TOKEN")
    #    )

        return model, processor







