import torch
from torch import nn
import torchvision.transforms as transforms
import numpy as np

# A variant of ResNet
class BasicBlock(nn.Module):
	def __init__(self, in_channels, out_channels, stride=1):
		super(BasicBlock, self).__init__()
		self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(out_channels)
		self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(out_channels)

		self.shortcut = nn.Sequential()
		if stride != 1 or in_channels != out_channels:
			self.shortcut = nn.Sequential(
				nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(out_channels)
			)

	def forward(self, x):
		out = nn.ReLU(inplace=True)(self.bn1(self.conv1(x)))
		out = self.bn2(self.conv2(out))
		out += self.shortcut(x)
		out = nn.ReLU(inplace=True)(out)
		return out

class ResNet(nn.Module):
	def __init__(self, num_classes=62):
		super(ResNet, self).__init__()
		self.in_channels = 16

		self.model = nn.Sequential(
		nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False),
		nn.BatchNorm2d(16),
		nn.ReLU(inplace=True),
		nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
		self._make_layer(16, 2, stride=1),
		self._make_layer(32, 2, stride=2),
		self._make_layer(64, 2, stride=2),
		self._make_layer(128, 2, stride=2),
		nn.AdaptiveAvgPool2d((1, 1)),
		nn.Flatten(start_dim=1),
		nn.Linear(128, num_classes)
		)
		self._initialize_weights()

	def _make_layer(self, out_channels, num_blocks, stride):
		strides = [stride] + [1] * (num_blocks - 1)
		layers = []
		for stride in strides:
			layers.append(BasicBlock(self.in_channels, out_channels, stride))
			self.in_channels = out_channels
		return nn.Sequential(*layers)

	def forward(self, x):
		return self.model(x)

	def _initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.Linear):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				nn.init.constant_(m.bias, 0)

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
	
def resnet_emnist():
	model = ResNet()
	state_dict = torch.load('model_weights/emnist_resnet_tuned.pth', map_location=torch.device('cpu'))
	model.load_state_dict(state_dict)
	model.eval()
	return model