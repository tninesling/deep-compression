# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

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
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def train(self):
      criterion = nn.CrossEntropyLoss()
      optimizer = optim.Adam(self.parameters(), lr=0.001)

      # Train the models
      for epoch in range(5):
          for i, (images, labels) in enumerate(train_loader):
              optimizer.zero_grad()
              outputs = self(images)
              loss = criterion(outputs, labels)
              loss.backward()
              optimizer.step()

#LeNet300 network
class LeNet300(nn.Module):
    def __init__(self):
        super(LeNet300, self).__init__()
        self.fc1 = nn.Linear(28*28, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, 10)
        self.relu = nn.ReLU()



    def forward(self, x):
        x = x.view(-1, 28*28)  # Flatten the input images
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def train(self):
      criterion = nn.CrossEntropyLoss()
      optimizer = optim.Adam(self.parameters(), lr=0.001)
      total_step = len(train_loader)
      for epoch in range(10):
          for i, (images, labels) in enumerate(train_loader):
              outputs = self(images)
              loss = criterion(outputs, labels)
              optimizer.zero_grad()
              loss.backward()
              optimizer.step()

if __name__ == "__main__":
  model5 = LeNet5()
  model300 = LeNet300()

  #get dataset and normalize
  transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.1307,), (0.3081,))
  ])

  train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
  test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

  train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
  test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

  #train models
  model5.train()
  model300.train()

  #evaluate models
  correct5 = 0
  correct300 = 0
  total = 0
  with torch.no_grad():
      for inputs, labels in test_loader:
          outputs5 = model5(inputs)
          outputs300 = model300(inputs)
          _, predicted5 = torch.max(outputs5.data, 1)
          _, predicted300 = torch.max(outputs300.data, 1)
          total += labels.size(0)
          correct5 += (predicted5 == labels).sum().item()
          correct300 += (predicted300 == labels).sum().item()

  print('Top1 error on LeNet5: %f %%' % (100 * (1 - (correct5 / total))))
  print('Top1 error on LeNet300: %f %%' % (100 * (1 - (correct300 / total))))
