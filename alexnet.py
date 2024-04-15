import yaml
import time
import torch
import torchvision
from config import Config
from kaggle_imagenet import KaggleImageNetDataset
from torch.utils.data import DataLoader


def nonempty_intersection(tensor1, tensor2):
    intersection_mask = torch.eq(tensor1.unsqueeze(1), tensor2)
    return intersection_mask.any()


class AlexNet:
    def __init__(self, config: Config):
        self.config = config
        self.model = torchvision.models.alexnet(
            weights=torchvision.models.AlexNet_Weights.IMAGENET1K_V1
        )
        self.model.eval()
        self.model.to(config.runtime.device)

    def predict(self, images):
        outputs = self.model(images)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        _, top_idx = torch.topk(probabilities, self.config.runtime.num_predictions)
        return top_idx


class Evaluator:
    def __init__(self, config: Config):
        self.config = config
        self.correct_predictions = 0
        self.total_images = 0

    def evaluate(self, dataloader: DataLoader, alexnet: AlexNet):
        print("Evaluating model...")
        self.correct_predictions = 0
        self.total_images = 0
        with torch.no_grad():
            for image_batch, label_batch in dataloader:
                image_batch = image_batch.to(self.config.runtime.device)
                label_batch = label_batch.to(self.config.runtime.device)
                prediction_batch = alexnet.predict(image_batch)

                for predictions, labels in zip(prediction_batch, label_batch):
                    self.total_images += 1
                    if nonempty_intersection(predictions, labels):
                        self.correct_predictions += 1

                    if self.should_report_progress():
                        self.report_progress(dataloader)

    def should_report_progress(self):
        ten_percent = len(dataloader.dataset) / 10
        return self.total_images % ten_percent == 0

    def report_progress(self, dataloader: DataLoader):
        progress_percent = self.total_images / len(dataloader.dataset) * 100
        accuracy = self.correct_predictions / self.total_images * 100
        print(
            f"{progress_percent:.1f}% complete, {accuracy:.1f}% accurate (top-{self.config.runtime.num_predictions})"
        )


class TimedEvaluator(Evaluator):
    def __init__(self, config):
        super().__init__(config)

    def evaluate(self, dataloader: DataLoader, alexnet: AlexNet):
        self.start_time = time.time()
        super().evaluate(dataloader, alexnet)
        print(f"Executed in {time.time() - self.start_time:.1f}s")


if __name__ == "__main__":
    config = Config()
    print(yaml.dump(config))

    imagenet = KaggleImageNetDataset(config)
    dataloader = DataLoader(
        imagenet,
        batch_size=config.runtime.batch_size,
        num_workers=config.runtime.num_workers,
    )
    alexnet = AlexNet(config)

    evaluator = TimedEvaluator(config)
    evaluator.evaluate(dataloader, alexnet)
