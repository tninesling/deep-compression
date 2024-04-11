import os
import torch
from dotenv import load_dotenv
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, Subset

load_dotenv()
imagenet_annotations_dir = os.getenv("IMAGENET_ANNOTATIONS_DIR")
imagenet_data_dir = os.getenv("IMAGENET_DATA_DIR")


class KaggleImageNetDataset(Dataset):
    def __init__(self, annotations_path, data_path):
        self.annotations_path = annotations_path
        self.data_path = data_path
        self.img_names = os.listdir(data_path)
        self.img_names.sort()
        self.transforms = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(
                    224  # TODO: Verify this is correct crop size for AlexNet
                ),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, i):
        img_path = os.path.join(self.data_path, self.img_names[i])
        img = Image.open(img_path).convert("RGB")
        # TODO: Get label from annotations file
        return self.transforms(img), ""


imagenet = KaggleImageNetDataset(
    imagenet_annotations_dir,
    imagenet_data_dir,
)
imagenet_sample = Subset(imagenet, list(range(10)))
dataloader = DataLoader(imagenet_sample, batch_size=16)

print("Download AlexNet model...")
alexnet = torch.hub.load("pytorch/vision:v0.10.0", "alexnet", pretrained=True)
alexnet.eval()

print("Evaluating model...")
with torch.no_grad():
    for images, _labels in dataloader:
        outputs = alexnet(images)
        print(f"{outputs}")
