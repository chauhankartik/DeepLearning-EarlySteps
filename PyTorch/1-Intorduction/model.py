# external libraries
import torch
import config
import torch.nn as nn

#define model
class NeuralNetwork(nn.Module):
	
	def __init__(self):
		super(NeuralNetwork, self).__init__()
		self.flatten = nn.Flatten()
		self.linear_relu_stack = nn.Sequential(
			nn.Linear(28*28, config.hidden_size),
			nn.ReLU(),
			nn.Linear(config.hidden_size, config.hidden_size),
			nn.ReLU(),
			nn.Linear(config.hidden_size, 10),
			nn.ReLU()
			)

	def forward(self, x):
		x = self.flatten(x)
		logits = self.linear_relu_stack(x)
		return logits

def newModel():
	Model = NeuralNetwork()
	return Model