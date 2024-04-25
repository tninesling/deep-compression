import numpy as np
import os
import yaml
import torch
from config import Config
from evaluator import TimedEvaluator
from kaggle_imagenet import KaggleImageNetDataset
from torch.quantization import MinMaxObserver, MovingAverageMinMaxObserver, MovingAveragePerChannelMinMaxObserver, PerChannelMinMaxObserver, QConfig, QuantStub, DeQuantStub
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.models import AlexNet, AlexNet_Weights


# PyTorch quantization does not yet support cuda. Before running,
# be sure to set DEVICE=cpu in your .env
class QuantizedAlexNet(AlexNet):
    def __init__(self):
        super().__init__()
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x):
        out = self.quant(x)
        out = super().forward(out)
        out = self.dequant(out)
        return out

    def quantize(self, weight_observer):
        self.qconfig = QConfig(
            activation=torch.quantization.default_observer,
            weight=weight_observer)
        torch.quantization.prepare(self, inplace=True)

    def calibrate(self, data):
        with torch.no_grad():
            for image_batch, _ in data:
                image_batch = image_batch.to(config.runtime.device)
                self(image_batch)

    def freeze(self):
        torch.quantization.convert(alexnet, inplace=True)

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

    observers = [
        MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_affine),
        MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric),
        MovingAverageMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_affine),
        PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric), 
        PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_affine),
        MovingAveragePerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric), 
        MovingAveragePerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_affine),
    ]

    for observer in observers:
        print(f"Quantizing with qscheme: {observer}")
        alexnet = QuantizedAlexNet()
        alexnet.load_state_dict(AlexNet_Weights.IMAGENET1K_V1.get_state_dict())
        alexnet.to(config.runtime.device)

        print("Checking model size...")
        alexnet.print_size()

        print("Quantizing AlexNet...")
        alexnet.quantize(observer)
        alexnet.calibrate(subset)
        alexnet.freeze()

        print("Checking post-quantization model size...")
        alexnet.print_size()

        print("Evaluating AlexNet...")
        evaluator = TimedEvaluator(config)
        evaluator.evaluate(dataloader, alexnet)
