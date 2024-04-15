import yaml
import torchvision
from config import Config
from evaluator import TimedEvaluator
from kaggle_imagenet import KaggleImageNetDataset
from torch.utils.data import DataLoader

if __name__ == "__main__":
    config = Config()
    print(yaml.dump(config))

    imagenet = KaggleImageNetDataset(config)
    dataloader = DataLoader(
        imagenet,
        batch_size=config.runtime.batch_size,
        num_workers=config.runtime.num_workers,
    )

    print("Evaluating AlexNet...")
    alexnet = torchvision.models.alexnet(
        weights=torchvision.models.AlexNet_Weights.IMAGENET1K_V1
    )
    evaluator = TimedEvaluator(config)
    evaluator.evaluate(dataloader, alexnet)
