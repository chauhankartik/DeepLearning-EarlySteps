#external libraries
import numpy as np 
import pickle
import os
import json
import torch
import model
import torch.utils.data as data
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.nn as nn

#internal utilities
import config
from model import NeuralNetwork, newModel
import data_loader
import test

#hyper-parameter setup
hyper_params = {
	"num_epochs" : config.num_epochs,
	"batch_size" : config.batch_size,
	"learning_rate" : config.learning_rate,
	"hidden_size" : config.hidden_size,
	"pretrained" : config.pretrained
}


# define a path to save experiment logs
experiment_path = "./{}".format(config.exp)
if not os.path.exists(experiment_path):
    os.mkdir(experiment_path)

#create data loaders
train_dataloader = data_loader.train_data_loader()
test_dataloader = data_loader.test_data_loader()

Model = model.newModel()
Model.to(config.device)

#define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adadelta(Model.parameters(), lr = config.learning_rate)

def train(dataloader, model, loss_fn, optimizer):
	size = len(dataloader.dataset)
	for batch, (X, y) in enumerate(dataloader):
		X, y = X.to(config.device), y.to(config.device)

		#compute prediction error
		pred = model(X)
		loss = loss_fn(pred, y)

		#back-prop
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if batch % 100 == 0:
			loss, current = loss.item(), batch * len(X)
			print(f"loss :{loss:>7f} [{current:>5d}/{size:>5d}]")


epoch = config.num_epochs
for t in range(epoch):
	print(f"Epochs {t+1}\n-----------------")
	train(train_dataloader, Model, criterion, optimizer)
	test.test(test_dataloader, Model, criterion)
print("Done")


