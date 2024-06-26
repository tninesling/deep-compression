import csv
import numpy as np
import os
import prunings
import quantizations
import torch
import pandas as pd
from config import Config
from copy import deepcopy
from dataloaders import mnist_loader
from models import leNet5, leNet300, alexnet, vgg16
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
    dataloaders = [
        mnist_loader,
        mnist_loader,
        # kaggle_imagenet_loader,
        # kaggle_imagenet_loader,
    ]
    models = [leNet5, leNet300]
    prunings = [
        prunings.no_prune,
        prunings.l1_unstructured_prune,
        prunings.l1_structured_prune_one_percent,
        prunings.l1_structured_prune_five_percent,
        prunings.random_unstructured_prune,
    ]
    quantizations = [
        quantizations.no_quantize,
        quantizations.quanto_int4_quantize,
        quantizations.quanto_int8_quantize,
        quantizations.quanto_float8_quantize,
    ]
    results = []

    for model, dataloader in zip(models, dataloaders):
        calibration_data = get_calibration_data(config, dataloader)

        for prune in prunings:
            for quantize in quantizations:
                case = f"{model.__name__}_{prune.__name__}_{quantize.__name__}"

                try:
                    print(f"Case: {case}")
                    candidate = deepcopy(model)
                    candidate.to(config.runtime.device)
                    print("Pruning model...")
                    candidate = prune(candidate)
                    print("Quantizing model...")
                    candidate = quantize(
                        candidate,
                        calibration_data,
                    )

                    print("Beginning evaluation...")
                    size = get_size_mb(candidate)
                    print(f"Model size: {size:.3f} MB")
                    evaluator = Evaluator(config)
                    [
                        total_images,
                        correct_predictions_top1,
                        correct_predictions_top5,
                        execution_time,
                    ] = evaluator.evaluate(dataloader, candidate)
                    results.append(
                        [
                            model.__name__,
                            prune.__name__,
                            quantize.__name__,
                            size,
                            total_images,
                            correct_predictions_top1,
                            correct_predictions_top5,
                            correct_predictions_top1 / total_images * 100,
                            correct_predictions_top5 / total_images * 100,
                            execution_time,
                        ]
                    )
                    print(
                        f"Evaluation complete. Accuracy {correct_predictions_top5 / total_images * 100:.1f}% (top5), time {execution_time:.3f}s"
                    )
                except Exception as e:
                    print(f"{case} failed: {e}")

    with open("results.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "model",
                "pruning",
                "quantization",
                "size (MB)",
                "total",
                "correct (top1)",
                "correct (top5)",
                "accuracy % (top1)",
                "accuracy % (top5)",
                "time (s)",
            ]
        )
        writer.writerows(results)

# df = pd.read_csv("results.csv", delimiter=",", engine="python")
# print(df.columns)

# [leNet5_original_size, leNet300_original_size, alexnet_original_size, vgg16_original_size] = df.query('pruning == "no_prune" and quantization == "no_quantize"')['size (MB)'].values
# leNet5_sizes= df.query('model == "lenet5"')['size (MB)']
# print(leNet5_sizes)
# lenet5_compressions = leNet5_sizes/leNet5_original_size

# print(lenet5_compressions)

# leNet300_results = df.query('model == "leNet300"')

# alexnet_results = df.query('model == "alexnet"')

# vgg16_results = df.query('model == "vgg16"')
