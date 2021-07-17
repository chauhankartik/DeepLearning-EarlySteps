# imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# Network Architecture
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, num_classes) -> None:
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x);

#set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model-parameter
input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 10

# load/download data
train_data = datasets.MNIST(root=r'./Data',
                            train = True,
                            transform = transforms.ToTensor(),
                            download = True)

train_loader = DataLoader(dataset = train_data, 
                          batch_size = batch_size,
                          shuffle = True)

test_data = datasets.MNIST(root=r'./Data',
                            train = False,
                            transform = transforms.ToTensor(),
                            download = True)

test_loader = DataLoader(dataset = test_data, 
                          batch_size = batch_size,
                          shuffle = True)

# initialize nn
model = NeuralNetwork(input_size=input_size, num_classes=num_classes).to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

# Train network

for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        data = data.reshape(data.shape[0], -1)

        # forward
        scores = model(data)
        loss = criterion(scores, target)

        # backward
        optimizer.zero_grad() 
        loss.backward()

        # gradient descent adam step
        optimizer.step()

def check_accuracy(loader, model):
    if loader.dataset.train:
        print('Checking accuracy on training data\n')
    else :
        print('Checking accuraacy on test data\n')
    num_correct = 0
    num_sample = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            x.reshape(x.shape[0], -1)

            scores = model(x) # 64 * 10
            _, predictions  = scores.max(1)
            num_correct += (predictions == y).sum()
            num_sample += predictions.size(0)

        print(f'Got {num_correct} / {num_sample} with accuracy')

check_accuracy(train_loader, model)
check_accuracy(test_loader, model)
