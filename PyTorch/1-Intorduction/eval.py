import config
import torch
import torch.nn as nn
import torch.utils.data as data
import os
import model
import pickle
import data_loader

#create data loaders
train_dataloader = data_loader.train_data_loader()
test_dataloader = data_loader.test_data_loader()

#loading model
model.Model.to(config.device)

#define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adadelta(model.Model.parameters(), lr = config.learning_rate)

epoch = config.num_epochs
for t in range(epoch):
	print(f"Epochs {t+1}\n-----------------")
	train.train(train_dataloader, model.Model, criterion, optimizer)
	test.test(test_dataloader, model.Model, criterion)
print("Done")