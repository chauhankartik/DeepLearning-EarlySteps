# external libraries
import torch
import torch.nn as nn

#define model
class NeuralNetwork(nn.Module):
	
	def __init__(self):
		super(NeuralNetwork, self).__init__()
		self.flatten = nn.Flatten()
		self.linear_relu_stack = nn.Sequential(
			nn.Linear(32 * 32, 512),
			nn.ReLU(),
			nn.Linear(512, 512),
			nn.ReLU(),
			nn.Linear(512, 10),
			nn.ReLU()
			)

	def forward(self, x):
		x = self.flatten(x)
		logits = self.linear_relu_stack(x)
		return logits