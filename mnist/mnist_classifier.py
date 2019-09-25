
import torch.nn as nn
import torch.nn.functional as F



class MnistNet(nn.Module):
	def __init__(self):
		super(MnistNet, self).__init__()
		self.conv1 = nn.Conv2d(1, 9, 3)
		self.conv2 = nn.Conv2d(9, 15, 3)
		self.lin1 = nn.Linear(375, 20)
		self.lin2 = nn.Linear(20, 10)
	def forward(self, x):
		out = F.max_pool2d(F.softplus(self.conv1(x)), kernel_size=2)
		out = F.max_pool2d(F.softplus(self.conv2(out)), kernel_size=2)
		out = self.lin2(F.softplus(self.lin1(out.view(out.size(0), -1))))
		return out
