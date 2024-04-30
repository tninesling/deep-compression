import os
import torch
from dotenv import load_dotenv


class Config:
    def __init__(self):
        load_dotenv()
        self.imagenet = ImageNetConfig()
        self.runtime = RuntimeConfig()


class ImageNetConfig:
    def __init__(self):
        self.annotations_dir = os.getenv("IMAGENET_ANNOTATIONS_DIR")
        self.data_dir = os.getenv("IMAGENET_DATA_DIR")
        self.synset_file = os.getenv("IMAGENET_SYNSET_FILE")


class RuntimeConfig:
    def __init__(self):
        self.batch_size = int(os.getenv("BATCH_SIZE"))
        self.device = torch.device(
            os.getenv("DEVICE") or "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.num_predictions = int(os.getenv("NUM_PREDICTIONS"))
        self.num_workers = int(os.getenv("NUM_WORKERS") or os.cpu_count())

config = Config()