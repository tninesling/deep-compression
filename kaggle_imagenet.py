import os
import torch
import torch.nn.functional as F
import torchvision
import xml.etree.ElementTree as Xml
from config import Config
from PIL import Image
from torch.utils.data import Dataset

PADDING_VALUE = -1


class Annotation:
    def __init__(self, config: Config, file_path, synsets: list):
        self.config = config
        self.synsets = synsets
        self.labels = set()

        xml = Xml.parse(file_path).getroot()
        filename = xml.find("filename").text
        if not file_path.endswith(f"{filename}.xml"):
            raise Exception("File name mismatch with annotation")

        for object in xml.iter("object"):
            name = object.find("name").text
            if name is not None:
                self.labels.add(name)

            if len(self.labels) == config.runtime.num_predictions:
                break

    def synset_idx_tensor(self):
        labels = torch.tensor([self.synsets.index(label) for label in self.labels])
        # Pad the tensor so all label tensors are the same size as the number of predictions.
        # The fact that the match the prediction tensor size is somewhat arbitrary, but all
        # label tensors must be the same size, regardless of what that size is.
        return F.pad(
            labels,
            (0, self.config.runtime.num_predictions - len(labels)),
            mode="constant",
            value=PADDING_VALUE,
        )


class KaggleImageNetDataset(Dataset):
    def __init__(self, config: Config):
        self.config = config
        self.img_names = os.listdir(config.imagenet.data_dir)
        self.img_names.sort()
        self.transforms = torchvision.models.AlexNet_Weights.IMAGENET1K_V1.transforms()
        self.synsets = []
        with open(config.imagenet.synset_file) as synset_file:
            for line in synset_file.readlines():
                [synset_id, _] = line.split(" ", 1)
                self.synsets.append(synset_id)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, i):
        img_path = os.path.join(self.config.imagenet.data_dir, self.img_names[i])
        img = Image.open(img_path).convert("RGB")
        annotation_path = os.path.join(
            self.config.imagenet.annotations_dir,
            self.img_names[i].replace("JPEG", "xml"),
        )
        annotation = Annotation(self.config, annotation_path, self.synsets)

        return self.transforms(img), annotation.synset_idx_tensor()
