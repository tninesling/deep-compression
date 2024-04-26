from torchvision.models import alexnet, AlexNet_Weights, vgg16, VGG16_Weights

alexnet = alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
alexnet.__name__ = "alexnet"

vgg16 = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
vgg16.__name__ = "vgg16"
