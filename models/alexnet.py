import torch
from torch import nn
from torch.nn import init
import torchvision.transforms as transforms
import numpy as np

# A variant of AlexNet
class AlexNet(nn.Module):
	def __init__(self, num_classes=62):
		super(AlexNet, self).__init__()
		self.features = nn.Sequential(
			nn.Conv2d(1, 16, kernel_size=3, padding=2),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=2),
			nn.Conv2d(16, 48, kernel_size=3, padding=2),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=2),
			nn.Conv2d(48, 96, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(96, 64, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(64, 64, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=2),
		)
		self.avgpool = nn.AdaptiveAvgPool2d((3, 3))
		self.classifier = nn.Sequential(
			nn.Dropout(),
			nn.Linear(64 * 3 * 3, 1024),
			nn.BatchNorm1d(1024),
			nn.ReLU(inplace=True),
			nn.Dropout(),
			nn.Linear(1024, 1024),
			nn.BatchNorm1d(1024),
			nn.ReLU(inplace=True),
			nn.Linear(1024, num_classes),
		)
		self._initialize_weights()

	def forward(self, x):
		x = self.features(x)
		x = self.avgpool(x)
		x = torch.flatten(x, 1)
		x = self.classifier(x)
		return x

	def _initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				if m.bias is not None:
					init.constant_(m.bias, 0)
			elif isinstance(m, nn.BatchNorm2d):
				init.constant_(m.weight, 1)
				init.constant_(m.bias, 0)
			elif isinstance(m, nn.Linear):
				init.normal_(m.weight, 0, 0.01)
				init.constant_(m.bias, 0)

	def evaluate(self, testX: np.ndarray, testY: np.ndarray, verbose):

		transform = transforms.Compose([
			transforms.Normalize((0.1736,), (0.3248,))  # Standardize
		])
		
		correct = 0
		for i in range(len(testX)):
			img = torch.tensor(testX[i]).permute(2, 0, 1)
			img = transform(img)
			img = img.unsqueeze(0)

			output = self.forward(img)
			predicted = torch.argmax(output).item()

			if testY[i][predicted] == 1:
				correct += 1

		accuracy = correct / len(testX)

		return None, accuracy

def alexnet():
	model = AlexNet()
	state_dict = torch.load('model_weights/emnist_alexnet_tuned.pth', map_location=torch.device('cpu'))
	model.load_state_dict(state_dict)
	model.eval()
	return model