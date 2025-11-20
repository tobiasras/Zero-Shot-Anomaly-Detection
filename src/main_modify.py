from data.data_loader import DatasetLoader
from models.load_vision_transformer import get_vit_model
import random
import torch
from util.logger import log
import numpy as np
from sklearn.metrics import roc_auc_score
from util.paths import PROJECT_ROOT
from data.transform import Transform
from util.visualization import plot_sim_matrix, save_img, plot_roc_curve
from scipy.signal import correlate  # Import circular correlation function

if __name__ == '__main__':
    log.info('Starting Zero-Shot Anomaly')

    seed = 42
    log.info('Setting seed: ' + str(seed))
    random.seed(seed)

    object_type = "grid"
    log.info('Object Type : ' + str(object_type))
    test_path = PROJECT_ROOT / 'data' / 'mvtec_anomaly_detection' / object_type / 'test'
    ref_path = PROJECT_ROOT / 'data' / 'mvtec_anomaly_detection' / object_type / 'train'

    # Used for transforming images to dinoV3
    image_transformer = Transform(256)
    test_dataset = DatasetLoader(test_path, image_transformer.get_transform())
    ref_dataset = DatasetLoader(ref_path, image_transformer.get_transform())

    ref_count = 50
    indices = random.sample(range(len(ref_dataset)), ref_count)
    ref_images_tuple = [ref_dataset[i] for i in indices]
    ref_images, _ = zip(*ref_images_tuple)

    log.info('Total ref images: ' + str(len(ref_images)))

    ref_images_stack = torch.stack(ref_images)

    index = 0

    model_name = "dino_vits16"
    log.info('Using model: ' + model_name)

    model = get_vit_model(model_name).eval()
    with torch.no_grad():
        ref_embed = model.get_intermediate_layers(ref_images_stack, n=3)
        ref_embed = torch.cat([layer[:, 1:, :] for layer in ref_embed], dim=-1)
        ref_embed = ref_embed.mean(dim=0) # Average across reference images

        # Normalize reference embeddings
        ref_embed = torch.nn.functional.normalize(ref_embed, p=2, dim =-1)

        scores = []
        labels = []
        for img, path in test_dataset:
            embed = model.get_intermediate_layers(img.unsqueeze(0), n=1)[0][:, 1:, :]

            # Calculate circular correlation instead of direct distance
            sim_matrix = []
            for ref_patch in ref_embed:  # Iterate over reference patches
                # Ensure both are 1D arrays
                correlation = correlate(embed[0].cpu().numpy().flatten(), ref_patch.cpu().numpy().flatten(), mode='full')
                sim_matrix.append(correlation)

            sim_matrix = np.array(sim_matrix)  # Convert to numpy array

            # Percentile-based scoring instead of just max
            percentile = 100
            score = - np.percentile(sim_matrix.max(axis=1), percentile)  # Use max correlation as score

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