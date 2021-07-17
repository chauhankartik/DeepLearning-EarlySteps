#imports 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# hypter-parameters
batch_size  = 200
epochs  = 20
num_classes = 10
learning_rate = 0.001


# load/download data
train_dataset = datasets.MNIST(
    root = './data',
    train = True,
    transform = transforms.ToTensor(),
    download = True
)

test_dataset = datasets.MNIST(
    root = './data',
    train = False,
    transform = transforms.ToTensor(),
    download = True
)

# wrapping data in DataLoader (a.k.a making data Iterable)
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False
)

# Network Architecture
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
    
        # convolution 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=15, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()
        # maxpool 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        # convolution 2
        self.conv2 = nn.Conv2d(in_channels=15, out_channels=40, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.ReLU()
        # maxpool 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        # dense layer
        self.dense1 = nn.Linear(40 * 7 * 7, 10)


    def forward(self, x):
        # convolution 1
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.maxpool1(out)
        # convolution 2
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.maxpool2(out)

        # resize 
        out = out.reshape(data.shape[0], -1)
        # dense layer
        out = self.dense1(out)

        return out

#set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# initialize nn
model = CNNModel().to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = learning_rate)

# training loop
iter = 0
for epoch in range(epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)

        # forward
        scores = model(data)
        loss = criterion(scores, target)

        # backward
        optimizer.zero_grad() 
        loss.backward()
        iter += 1

        # gradient descent adam step
        optimizer.step()

        if iter % 500 == 0:
            # Calculate Accuracy         
            correct = 0
            total = 0
            # Iterate through test dataset
            for data, target in test_loader:
                # Load images
                data = data.to(device)
                # Forward pass only to get logits/output
                target = target.to(device)

                outputs = model(data)

                # Get predictions from the maximum value
                _, predicted = torch.max(outputs.data, 1)

                # Total number of labels
                total += target.size(0)

                # Total correct predictions
                correct += (predicted == target).sum()

            accuracy = 100 * correct / total

            # Print Loss
            print('Iteration: {}. Loss: {}. Accuracy: {}'.format(iter, loss.item(), accuracy))


# save trained model
PATH = './mnist_net.pth'
torch.save(model.state_dict(), PATH)



