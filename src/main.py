from data.data_loader import DatasetLoader
from models.load_vision_transformer import get_vit_model
import random
import torch
from util.logger import log
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_auc_score
from util.paths import PROJECT_ROOT
from data.transform import Transform
from evaluation.distance import calculate_distance
from util.visualization import plot_sim_matrix, save_img, plot_roc_curve
import argparse
import json

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", required=True, help="Path to JSON config file")
    return parser.parse_args()

def load_config(path):
    with open(path, "r") as f:
        config = json.load(f)
    return config


if __name__ == '__main__':
    log.info('Starting Zero-Shot Anomaly')
    args = read_args()
    config = load_config(args.config)

    seed = config['seed']
    log.info('Setting seed: ' + str(seed))
    random.seed(seed)

    object_type = "wood"
    log.info('Object Type : ' + str(object_type))
    test_path = PROJECT_ROOT / 'data' / 'mvtec_anomaly_detection' / object_type / 'test'
    ref_path = PROJECT_ROOT / 'data' / 'mvtec_anomaly_detection' / object_type / 'train'

    # used for transforming images to dinoV3
    image_transformer = Transform(config['image_size'])
    test_dataset = DatasetLoader(test_path, image_transformer.get_transform())
    ref_dataset = DatasetLoader(ref_path, image_transformer.get_transform())

    ref_count = config['ref_img_count']
    indices = random.sample(range(len(ref_dataset)), ref_count)
    ref_images_tuple = [ref_dataset[i] for i in indices]
    ref_images, _ = zip(*ref_images_tuple)

    log.info('total ref images: ' + str(len(ref_images)))

    ref_images_stack = torch.stack(ref_images)
    model_name = "dino_vits16"
    log.info('Using model: ' + model_name)

    model = get_vit_model(model_name).eval()

    index = 0
    with torch.no_grad():
        ref_embed = model.get_intermediate_layers(ref_images_stack, n=1)[0][:, 1:, :]
        ref_embed = ref_embed.mean(dim=0)

        scores = []
        labels = []
        for img, path in test_dataset:
            embed = model.get_intermediate_layers(img.unsqueeze(0), n=1)[0][:, 1:, :]
            sim_matrix = calculate_distance(ref_embed, embed[0])  # [num_patches]

            topk = 5
            score = -sim_matrix.topk(topk, largest=False).values.mean().item() # high sim = not anomaly

            scores.append((score, path))
            is_anomaly = "good" not in str(path)
            labels.append(1 if is_anomaly else 0)
            index += 1

    img = image_transformer.reverse_transform(ref_images_stack.squeeze(0).mean(dim=0))
    save_img(img, PROJECT_ROOT / 'experiments' / 'figures', "ref_images")

    values = np.array([s for s, _ in scores]).reshape(-1, 1)
    plot_roc_curve(labels, values)


    auc = roc_auc_score(labels, values)
    log.info(f"AUROC: {auc:.4f}")

