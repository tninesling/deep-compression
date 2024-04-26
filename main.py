import csv
import numpy as np
import os
import quantizations
import torch
from config import Config
from copy import deepcopy
from dataloaders import kaggle_imagenet_loader
from models import alexnet, vgg16
from prunings import no_prune
from evaluator import Evaluator
from torch.utils.data import DataLoader, SubsetRandomSampler


def get_size_mb(model):
    torch.save(model.state_dict(), "temp.p")
    size = os.path.getsize("temp.p") / 1e6
    os.remove("temp.p")
    return size


def get_calibration_data(config: Config, dataloader: DataLoader):
    dataset = dataloader.dataset
    subset_indices = np.random.choice(len(dataset), 1000, replace=False)
    subset_sampler = SubsetRandomSampler(subset_indices)
    return DataLoader(
        dataset,
        batch_size=config.runtime.batch_size,
        num_workers=config.runtime.num_workers,
        sampler=subset_sampler,
    )


if __name__ == "__main__":
    config = Config()
    dataloaders = [kaggle_imagenet_loader, kaggle_imagenet_loader]
    models = [alexnet] #, vgg16]
    prunings = [no_prune]
    quantizations = [
        # quantizations.no_quantize,
        quantizations.affine_minmax_per_channel,
        quantizations.affine_minmax_per_tensor,
        # quantizations.affine_moving_avg_per_channel,
        # quantizations.affine_moving_avg_per_tensor,
        # quantizations.symmetric_minmax_per_channel,
        # quantizations.symmetric_minmax_per_tensor,
        # quantizations.symmetric_moving_avg_per_channel,
    ]

    results = []

    for model, dataloader in zip(models, dataloaders):
        for prune in prunings:
            for quantize in quantizations:
                name = f"{model.__name__}_{prune.__name__}_{quantize.__name__}"
                calibration_data = get_calibration_data(config, dataloader)

                print(f"Case: {name}")
                candidate = deepcopy(model)
                candidate = prune(candidate)
                candidate = quantize(
                    candidate,
                    calibration_data,
                )

                print("Beginning evaluation...")
                size = get_size_mb(candidate)
                evaluator = Evaluator(config)
                [total_images, correct_predictions, accuracy, execution_time] = (
                    evaluator.evaluate(dataloader, candidate)
                )
                results.append(
                    [
                        name,
                        size,
                        total_images,
                        correct_predictions,
                        accuracy,
                        execution_time,
                    ]
                )
                print("Evaluation complete.")

    with open("results.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(results)
