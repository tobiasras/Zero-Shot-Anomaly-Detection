import io

from data.data_loader import DatasetLoader
from models.load_vision_transformer import get_vit_model
import random
import torch
from util.logger import log
import numpy as np
from sklearn.metrics import roc_auc_score
from util.paths import PROJECT_ROOT

from util.visualization import visualize_anomaly
from data.transform import Transform
from evaluation.distance import calculate_distance
import argparse
import json
import os

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", required=True, help="Path to JSON config file")
    return parser.parse_args()


def load_config(path):
    with open(path, "r") as f:
        config = json.load(f)
    return config


def run_experiment(object_type: str, experiment_param, vit_model, output_path):
    test_path = PROJECT_ROOT / 'data' / 'mvtec_anomaly_detection' / object_type / 'test'
    ref_path = PROJECT_ROOT / 'data' / 'mvtec_anomaly_detection' / object_type / 'train'

    # used for transforming images to dinoV3
    image_transformer = Transform(experiment_param['image_size'])
    test_dataset = DatasetLoader(test_path, image_transformer.get_transform())
    ref_dataset = DatasetLoader(ref_path, image_transformer.get_transform())

    # fetch ref images
    ref_count = experiment_param['ref_img_count']
    indices = random.sample(range(len(ref_dataset)), ref_count)

    # unpack images  ref_dataset gives: image, Path.
    ref_images_tuple = [ref_dataset[i] for i in indices]
    ref_images, _ = zip(*ref_images_tuple)

    ref_images_stack = torch.stack(ref_images)
    model = vit_model

    index = 0
    with torch.no_grad():
        # run ref images through dino model to get embeddings:
        ref_embed = model.get_intermediate_layers(ref_images_stack, n=1)[0][:, 1:, :]

        scores = []
        labels = []
        for img, path in test_dataset:
            embed = model.get_intermediate_layers(img.unsqueeze(0), n=1)[0][:, 1:, :]

            distance_type = experiment_param['distance']
            ref_aggregation_method = experiment_param['ref_aggregation_method']
            sim_matrix = calculate_distance(ref_embed, embed[0], measure_type=distance_type, ref_aggregation_method=ref_aggregation_method)  # [num_patches]

            topk = experiment_param["top_n"]
            if distance_type.lower() == "cosine":
                anomaly_score = 1 - sim_matrix
            else:
                anomaly_score = sim_matrix

            #visualize_anomaly(sim_matrix, path, topk, output_path, experiment_name)

            score = anomaly_score.topk(topk, largest=True).values.mean().item()

            scores.append((score, path))

            is_anomaly = "good" not in str(path)
            labels.append(1 if is_anomaly else 0)
            index += 1

    values = np.array([s for s, _ in scores]).reshape(-1, 1)
    auc = roc_auc_score(labels, values)

    log.info(f"{object_type}: AUROC: {auc:.4f}")

    return auc



if __name__ == '__main__':
    log.info('Starting Zero-Shot Anomaly')
    args = read_args()
    config = load_config(args.config)

    seed = config['seed']
    random.seed(seed)

    result = {}

    # each experiment declared:
    for experiment_param in config["experiments"]:
        experiment_name = f"{experiment_param['image_size']}_{experiment_param['vit_model']}_{experiment_param['distance']}_{experiment_param['ref_aggregation_method']}_{experiment_param['ref_img_count']}_{experiment_param['top_n']}"
        output_file = f"{config['output_path']}/{experiment_name}"
        # loop over dataset objects: bottle, cable, ects
        log.info(f'\n{experiment_param}')

        model_name = experiment_param['vit_model']
        model = get_vit_model(model_name).eval()

        data = {}
        all_scores = []
        for object_type in config['object']:
            score = run_experiment(object_type, experiment_param, model, output_file)
            all_scores.append(score)
            data[object_type] = score

        avg_score = np.mean(all_scores)
        data['all'] = avg_score
        log.info(f'Avg score: {avg_score}')

        os.makedirs(config['output_path'], exist_ok=True)
        with open(output_file + ".json", "w") as f:
            json.dump(data, f, indent=4)
