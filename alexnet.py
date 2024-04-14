import os
import time
import torch
import torch.nn.functional as F
import torchvision
import xml.etree.ElementTree as Xml
from dotenv import load_dotenv
from PIL import Image
from torch.utils.data import DataLoader, Dataset

load_dotenv()
imagenet_annotations_dir = os.getenv("IMAGENET_ANNOTATIONS_DIR")
imagenet_data_dir = os.getenv("IMAGENET_DATA_DIR")
imagenet_synset_file = os.getenv("IMAGENET_SYNSET_FILE")

BATCH_SIZE = 32
NUM_PREDICTIONS = 5
PADDING_VALUE = -1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_workers = os.cpu_count()
print(f"device: {device}, batch size: {BATCH_SIZE}, workers: {num_workers}")

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
        # Pad the tensor so all label tensors are the same size as the number of predictions.
        # The fact that the match the prediction tensor size is somewhat arbitrary, but all
        # label tensors must be the same size, regardless of what that size is.
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
        self.transforms = torchvision.models.AlexNet_Weights.IMAGENET1K_V1.transforms()

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, i):
        img_path = os.path.join(self.data_path, self.img_names[i])
        img = Image.open(img_path).convert("RGB")
        annotation_path = os.path.join(
            self.annotations_path, self.img_names[i].replace("JPEG", "xml")
        )
        annotation = Annotation(annotation_path)

        return self.transforms(img), annotation.synset_idx_tensor()


class AlexNet:
    def __init__(self):
        self.model = torchvision.models.alexnet(
            weights=torchvision.models.AlexNet_Weights.IMAGENET1K_V1
        )

    def eval(self):
        self.model.eval()

    def predict(self, images):
        outputs = self.model(images)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        _, top_idx = torch.topk(probabilities, NUM_PREDICTIONS)
        return top_idx

    def to(self, device):
        self.model.to(device)


def nonempty_intersection(tensor1, tensor2):
    intersection_mask = torch.eq(tensor1.unsqueeze(1), tensor2)
    return intersection_mask.any()


if __name__ == "__main__":
    imagenet = KaggleImageNetDataset(
        imagenet_annotations_dir,
        imagenet_data_dir,
    )
    dataloader = DataLoader(imagenet, batch_size=32, num_workers=num_workers)
    alexnet = AlexNet()

    print("Evaluating model...")
    alexnet.eval()
    alexnet.to(device)

    correct = 0
    total = 0
    last_checkpoint = 0
    start_time = time.time()
    with torch.no_grad():
        checkpoint_start_time = time.time()
        for image_batch, label_batch in dataloader:
            image_batch = image_batch.to(device)
            label_batch = label_batch.to(device)
            prediction_batch = alexnet.predict(image_batch)

            for predictions, labels in zip(prediction_batch, label_batch):
                total += 1
                if nonempty_intersection(predictions, labels):
                    correct += 1

            if total - last_checkpoint > 1000:
                print(
                    f"{total / len(imagenet) * 100:.1f}% complete. Time since last checkpoint: {time.time() - checkpoint_start_time:.1f}s"
                )
                last_checkpoint = total
                checkpoint_start_time = time.time()

    accuracy = correct / total * 100
    print(f"Top-{NUM_PREDICTIONS} accuracy: {accuracy:.1f}% ({correct} / {total})")
    print(f"Executed in {time.time() - start_time:.1f}s")
