#external libraries
import torch.utils.data as data
import config
from torchvision import datasets
from torchvision.transforms import ToTensor

data_dir = config.data_dir

#downloading training data
training_data = datasets.CIFAR10(
	root = data_dir,
	train = True,
	download = True,
	transform = ToTensor(),
	)

#downloading test data
test_data = datasets.CIFAR10(
	root = data_dir,
	train = False,
	download = True,
	transform = ToTensor(),
	)

#create data loaders
train_data_loader = data.DataLoader(training_data, batch_size = batch_size)
test_data_loader = data.DataLoader(test_data, batch_size = batch_size)



