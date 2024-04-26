from config import Config
from kaggle_imagenet import KaggleImageNetDataset
from torch.utils.data import DataLoader

_config = Config()
kaggle_imagenet_loader = DataLoader(
    KaggleImageNetDataset(_config),
    batch_size=_config.runtime.batch_size,
    num_workers=_config.runtime.num_workers,
)
