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
from utils import save_checkpoint, compute_batch_metrics

#hyper-parameter setup
hyper_params = {
	"num_epochs" : config.num_epochs,
	"batch_size" : config.batch_size,
	"learning_rate" : config.learning_rate,
	"hidden_size" : config.hidden_size,
	"cuda" : config.cuda,
	"pretrained" : config.pretrained
}

# train on GPU if CUDA variable is set to True (a GPU with CUDA is needed to do so)
device = torch.device("cuda" if hyper_params["cuda"] else "cpu")
torch.manual_seed(42)

# define a path to save experiment logs
experiment_path = "output/{}".format(config.exp)
if not os.path.exists(experiment_path):
    os.mkdir(experiment_path)

#loading data
train_dataloader = data_loader.train_dataloader
test_dataloader = data_loader.test_dataloader

#loading model
model = NeuralNetwork()
model.to(device)

#define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adadelta(model.parameters(), learning_rate = config.learning_rate)

def train(dataloader, model, loss_fn, optimizer):
	size = len(dataloader.dataset)
	for batch, (X, y) in enumerate(dataloader):
		X, y = X.to(device), y.to(device)

		#compute prediction error
		pred = model(X)
		loss_fn = loss_fn(pred, y)

		#back-prop
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if batch % 100 == 0:
			loss, current = loss.item(), batch * len(X)
			print(f"loss :{loss:>7f} [{current:>5d}/{size:>5d}]")



