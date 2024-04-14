import os
import torch
import torch.nn.functional as F
import xml.etree.ElementTree as Xml
from dotenv import load_dotenv
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, Subset

load_dotenv()
imagenet_annotations_dir = os.getenv("IMAGENET_ANNOTATIONS_DIR")
imagenet_data_dir = os.getenv("IMAGENET_DATA_DIR")
imagenet_synset_file = os.getenv("IMAGENET_SYNSET_FILE")

NUM_PREDICTIONS = 5
PADDING_VALUE = -1

synsets = []
with open(imagenet_synset_file) as synset_file:
    for line in synset_file.readlines():
        [synset_id, _] = line.split(" ", 1)
        synsets.append(synset_id)


class Annotation:
    def __init__(self, file_path):
        self.labels = set()

        xml = Xml.parse(file_path).getroot()
        filename = xml.find("filename").text
        if not file_path.endswith(f"{filename}.xml"):
            raise Exception("File name mismatch with annotation")

        for object in xml.iter("object"):
            name = object.find("name").text
            if name is not None:
                self.labels.add(name)

            if len(self.labels) == NUM_PREDICTIONS:
                break

    def synset_idx_tensor(self):
        labels = torch.tensor([synsets.index(label) for label in self.labels])
        return F.pad(
            labels,
            (0, NUM_PREDICTIONS - len(labels)),
            mode="constant",
            value=PADDING_VALUE,
        )


class KaggleImageNetDataset(Dataset):
    def __init__(self, annotations_path, data_path):
        self.annotations_path = annotations_path
        self.data_path = data_path
        self.img_names = os.listdir(data_path)
        self.img_names.sort()
        # Transforms from https://pytorch.org/hub/pytorch_vision_alexnet/
        self.transforms = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, i):
        img = Image.open(os.path.join(self.data_path, self.img_names[i])).convert("RGB")
        annotation = Annotation(
            os.path.join(
                self.annotations_path, self.img_names[i].replace("JPEG", "xml")
            )
        )

        return self.transforms(img), annotation.synset_idx_tensor()


class AlexNet:
    def __init__(self):
        self.model = torch.hub.load(
            "pytorch/vision:v0.10.0", "alexnet", pretrained=True
        )

    def eval(self):
        self.model.eval()

    def predict(self, images):
        outputs = self.model(images)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        _top_prob, top_idx = torch.topk(probabilities, NUM_PREDICTIONS)
        return top_idx

    def to(self, device):
        self.model.to(device)


imagenet = KaggleImageNetDataset(
    imagenet_annotations_dir,
    imagenet_data_dir,
)
imagenet_sample = Subset(imagenet, list(range(10)))
dataloader = DataLoader(imagenet_sample, pin_memory=True, batch_size=16)
alexnet = AlexNet()

print("Evaluating model...")
device = torch.device(
    "cpu"
)  # torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}")

alexnet.eval()
alexnet.to(device)

correct = 0
total = 0
with torch.no_grad():
    for images, labels in dataloader:
        images.to(device)
        labels.to(device)

        predictions = alexnet.predict(images)
        for label in labels:
            if label is PADDING_VALUE:
                continue

            if label in predictions:
                correct += 1

            total += 1

accuracy = correct / total * 100
print(f"Top-{NUM_PREDICTIONS} accuracy: {accuracy}% ({correct} / {total})")
