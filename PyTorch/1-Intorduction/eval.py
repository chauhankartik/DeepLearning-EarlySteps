import torch
import config
import train
import test
import data_loader
from model import NeuralNetwork

#loading data
train_dataloader = data_loader.train_dataloader
test_dataloader = data_loader.test_dataloader

#loading model
model = NeuralNetwork()
model.to(device)

#define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adadelta(model.parameters(), learning_rate = config.learning_rate)

epoch = config.num_epochs
for t in range(epochs):
	print(f"Epochs {t+1}\n-----------------")
	train.train(train_dataloader, model)
	test.test(test_dataloader, model, criterion)
print("Done")