from torchvision.models import alexnet, AlexNet_Weights, vgg16, VGG16_Weights

alexnet = alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
alexnet.__name__ = "alexnet"

vgg16 = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
vgg16.__name__ = "vgg16"

#LeNet5 network
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.reshape(-1, 16 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        
#LeNet300 network
class LeNet300(nn.Module):
    def __init__(self):
        super(LeNet300, self).__init__()
        self.fc1 = nn.Linear(28*28, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.reshape(-1, 28*28)  # Flatten the input images
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        
leNet300 = LeNet300()
leNet5 = LeNet5()

leNet300.load_state_dict(torch.load("trained_models/trainedModel300.p"))
leNet5.load_state_dict(torch.load("trained_models/trainedModel5.p"))
