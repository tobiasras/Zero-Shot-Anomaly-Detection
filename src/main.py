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


def run_experiment(object_type: str, experiment_param, vit_model, config):
    test_path = PROJECT_ROOT / 'data' / 'mvtec_anomaly_detection' / object_type / 'test'
    ref_path = PROJECT_ROOT / 'data' / 'mvtec_anomaly_detection' / object_type / 'train'

    # used for transforming images to dinoV3
    image_transformer = Transform(experiment_param['image_size'])
    test_dataset = DatasetLoader(test_path, image_transformer.get_transform())
    ref_dataset = DatasetLoader(ref_path, image_transformer.get_transform())

    ref_count = experiment_param['ref_img_count']
    indices = random.sample(range(len(ref_dataset)), ref_count)
    ref_images_tuple = [ref_dataset[i] for i in indices]
    ref_images, _ = zip(*ref_images_tuple)

    ref_images_stack = torch.stack(ref_images)

    model = vit_model
    index = 0
    with torch.no_grad():
        ref_embed = model.get_intermediate_layers(ref_images_stack, n=1)[0][:, 1:, :]
        scores = []
        labels = []
        for img, path in test_dataset:
            embed = model.get_intermediate_layers(img.unsqueeze(0), n=1)[0][:, 1:, :]

            distance_type = experiment_param['distance']
            sim_matrix = calculate_distance(ref_embed, embed[0], measure_type=distance_type)  # [num_patches]

            topk = experiment_param["top_n"]
            if distance_type.lower() == "cosine":
                score = -sim_matrix.topk(topk, largest=False).values.mean().item()
            else:
                score = sim_matrix.topk(topk, largest=True).values.mean().item()

            scores.append((score, path))
            is_anomaly = "good" not in str(path)
            labels.append(1 if is_anomaly else 0)
            index += 1

    values = np.array([s for s, _ in scores]).reshape(-1, 1)
    plot_roc_curve(labels, values)

    auc = roc_auc_score(labels, values)

    log.info(f"{object_type}: AUROC: {auc:.4f}")


if __name__ == '__main__':
    log.info('Starting Zero-Shot Anomaly')
    args = read_args()
    config = load_config(args.config)

    log.info('\n' + json.dumps(config, indent=4))

    seed = config['seed']
    random.seed(seed)

    # each experiment declared:
    for experiment_param in config["experiments"]:
        # loop over dataset objects: bottle, cable, ects
        log.info(f'\n{experiment_param}')

        model_name = experiment_param['vit_model']
        model = get_vit_model(model_name).eval()

        for object_type in config['object']:
            run_experiment(object_type, experiment_param, model, config)
