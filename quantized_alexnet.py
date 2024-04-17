import yaml
import torch
import torchvision
from config import Config
from evaluator import TimedEvaluator
from kaggle_imagenet import KaggleImageNetDataset
from torch.quantization import QuantStub, DeQuantStub
from torch.utils.data import DataLoader, Subset
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

if __name__ == "__main__":
    config = Config()
    print(yaml.dump(config))

    imagenet = KaggleImageNetDataset(config)
    dataloader = DataLoader(
        imagenet,
        batch_size=config.runtime.batch_size,
        num_workers=config.runtime.num_workers,
    )

    subset = DataLoader(Subset(imagenet, range(10)))

    alexnet = QuantizedAlexNet()
    alexnet.load_state_dict(AlexNet_Weights.IMAGENET1K_V1.get_state_dict())
    alexnet.to(config.runtime.device)

    print("Quantizing AlexNet...")
    alexnet.qconfig = torch.quantization.default_qconfig
    torch.quantization.prepare(alexnet, inplace=True)
    with torch.no_grad():
        for image_batch, _ in subset:
            image_batch = image_batch.to(config.runtime.device)
            alexnet(image_batch) # Throw away the predictions since it's just tuning the quantization
    
    torch.quantization.convert(alexnet, inplace=True)

    print("Evaluating AlexNet...")
    evaluator = TimedEvaluator(config)
    evaluator.evaluate(dataloader, alexnet)
