import time
import torch
from config import Config
from torch.utils.data import DataLoader


def nonempty_intersection(tensor1, tensor2):
    intersection_mask = torch.eq(tensor1.unsqueeze(1), tensor2)
    return intersection_mask.any()


class Evaluator:
    def __init__(self, config: Config):
        self.config = config
        self.correct_predictions = 0
        self.total_images = 0

    def evaluate(self, dataloader, model):
        model.eval()
        model.to(self.config.runtime.device)

        self.correct_predictions_top1 = 0
        self.correct_predictions_top5 = 0
        self.total_images = 0
        start_time = time.time()
        with torch.inference_mode():
            for image_batch, label_batch in dataloader:
                image_batch = image_batch.to(self.config.runtime.device)
                label_batch = label_batch.to(self.config.runtime.device)
                outputs = model(image_batch)
                _, prediction_batch = torch.topk(
                    outputs, 5, sorted=True
                )

                for predictions, labels in zip(prediction_batch, label_batch):
                    self.total_images += 1
                    if nonempty_intersection(predictions[0:1], labels):
                        self.correct_predictions_top1 += 1
                    if nonempty_intersection(predictions, labels):
                        self.correct_predictions_top5 += 1

                    if self.should_report_progress(dataloader):
                        self.report_progress(dataloader)

        return [
            self.total_images,
            self.correct_predictions_top1,
            self.correct_predictions_top5,
            time.time() - start_time,
        ]

    def should_report_progress(self, dataloader: DataLoader):
        ten_percent = len(dataloader.dataset) / 10
        return self.total_images % ten_percent == 0

    def report_progress(self, dataloader: DataLoader):
        progress_percent = self.total_images / len(dataloader.dataset) * 100
        print(f"{progress_percent:.1f}% complete")
