#external libraries
import numpy as np 
import pickle
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

#internal utilities
import config
from model import NeuralNetwork
import data_loader

#hyper-parameter setup
hyper_params = {
	"num_epochs" : config.num_epochs,
	"batch_size" : config.batch_size,
	"learning_rate" : config.learning_rate,
	"hidden_size" : config.hidden_size,
	"pretrained" : config.pretrained
}

# train on GPU if CUDA variable is set to True (a GPU with CUDA is needed to do so)


test_dataloader = data_loader.test_data_loader()

#define loss and optimizer
criterion = nn.CrossEntropyLoss()


def test(dataloader, model, loss_fn):
	optimizer = torch.optim.Adadelta(model.parameters(), lr = config.learning_rate)
	size = len(dataloader.dataset)
	model.eval()
	test_loss, correct = 0, 0
	with torch.no_grad():
		for X,y in dataloader:
			X,y = X.to(config.device), y.to(config.device)
			pred = model(X)
			test_loss += loss_fn(pred, y).item()
			correct += (pred.argmax(1) == y).type(torch.float).sum().item()

	test_loss /= size
	correct /= size
	print(f"Test error :\n Accuracy : {(100*correct) : >0.1f} %, Avg loss : {test_loss: >8f}\n")
