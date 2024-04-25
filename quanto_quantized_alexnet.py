import numpy as np
import os
import quanto
import torch
import yaml
from config import Config
from evaluator import TimedEvaluator
from kaggle_imagenet import KaggleImageNetDataset
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.models import AlexNet, AlexNet_Weights


# PyTorch quantization does not yet support cuda. Before running,
# be sure to set DEVICE=cpu in your .env
class QuantizedAlexNet(AlexNet):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        out = super().forward(x)
        if isinstance(out, quanto.QTensor):
            out = out.dequantize()
        return out

    def quantize(self):
        quanto.quantize(self, weights=quanto.qint8)

    def calibrate(self, data):
        with quanto.Calibration():
            for image_batch, _ in data:
                image_batch = image_batch.to(config.runtime.device)
                self(image_batch)

    def freeze(self):
        quanto.freeze(self)

    def print_size(self):
        torch.save(self.state_dict(), "temp.p")
        print(f"Size (MB): {os.path.getsize('temp.p') / 1e6}")
        os.remove("temp.p")


if __name__ == "__main__":
    config = Config()
    print(yaml.dump(config))

    imagenet = KaggleImageNetDataset(config)
    dataloader = DataLoader(
        imagenet,
        batch_size=config.runtime.batch_size,
        num_workers=config.runtime.num_workers,
    )

    subset_indices = np.random.choice(len(imagenet), 1000, replace=False)
    subset_sampler = SubsetRandomSampler(subset_indices)
    subset = DataLoader(
        imagenet,
        batch_size=config.runtime.batch_size,
        num_workers=config.runtime.num_workers,
        sampler=subset_sampler,
    )

    alexnet = QuantizedAlexNet()
    alexnet.load_state_dict(AlexNet_Weights.IMAGENET1K_V1.get_state_dict())
    alexnet.to(config.runtime.device)

    print("Checking model size...")
    alexnet.print_size()

    print("Quantizing AlexNet...")
    alexnet.quantize()
    alexnet.calibrate(subset)
    alexnet.freeze()

    print("Checking post-quantization model size...")
    alexnet.print_size()

    print("Evaluating AlexNet...")
    evaluator = TimedEvaluator(config)
    evaluator.evaluate(dataloader, alexnet)
