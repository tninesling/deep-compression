import torchvision
import torchvision.transforms as transforms
from config import Config
from kaggle_imagenet import KaggleImageNetDataset
from torch.utils.data import DataLoader

_config = Config()
kaggle_imagenet_loader = DataLoader(
    KaggleImageNetDataset(_config),
    batch_size=_config.runtime.batch_size,
    num_workers=_config.runtime.num_workers,
)

mnist_dataset = torchvision.datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    ),
)
mnist_loader = DataLoader(
    mnist_dataset,
    batch_size=_config.runtime.batch_size,
    num_workers=_config.runtime.num_workers,
    shuffle=False,
)
