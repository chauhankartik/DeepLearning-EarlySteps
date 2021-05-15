import config
import torch
import torch.nn as nn
import torch.utils.data as data
import model
import data_loader

#create data loaders
train_dataloader = data_loader.train_data_loader()
test_dataloader = data_loader.test_data_loader()

#define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adadelta(model.Model.parameters(), lr = config.learning_rate)

