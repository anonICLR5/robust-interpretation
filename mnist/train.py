'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from mnist_classifier import MnistNet

from utils import progress_bar


parser = argparse.ArgumentParser(description='PyTorch MNIST Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')

transform = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize((0.1307,), (0.3081,)),
])


# Model
print('==> Building model..')
net = MnistNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
net = net.cuda()
net = torch.nn.DataParallel(net)
cudnn.benchmark = True

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)

val_len = int(np.floor(.2 * len(trainset)))

if args.resume:
	# Load checkpoint.
	print('==> Resuming from checkpoint..')
	assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
	checkpoint = torch.load('./checkpoint/ckpt.t7')
	net.load_state_dict(checkpoint['net'])
	best_acc = checkpoint['acc']
	start_epoch = checkpoint['epoch']
	val_indices = checkpoint['val_indices']
else:
	val_indices = np.random.choice(list(range(len(trainset))), size=val_len, replace=False)

train_indices = list(set(range(len(trainset))) - set(val_indices))

trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, num_workers=2, sampler=torch.utils.data.sampler.SubsetRandomSampler(train_indices))

valloader = torch.utils.data.DataLoader(trainset, batch_size=256, num_workers=2, sampler=torch.utils.data.sampler.SubsetRandomSampler(val_indices))

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

# Training
def train(epoch):
	print('\nEpoch: %d' % epoch)
	net.train()
	train_loss = 0
	correct = 0
	total = 0
	for batch_idx, (inputs, targets) in enumerate(trainloader):
		inputs, targets = Variable(inputs).cuda(), Variable(targets).cuda()
		optimizer.zero_grad()
		outputs = net(inputs)
		loss = criterion(outputs, targets)
		loss.backward()
		optimizer.step()
		train_loss += loss.data
		_, predicted = outputs.max(1)
		total += targets.size(0)
		correct += predicted.eq(targets).sum().data

		progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
			% (train_loss/(batch_idx+1), 100.*float(correct)/total, correct, total))

def test(epoch):
	global best_acc
	net.eval()
	test_loss = 0
	correct = 0
	total = 0
	for batch_idx, (inputs, targets) in enumerate(valloader):
		inputs, targets = Variable(inputs).cuda(), Variable(targets).cuda()
		outputs = net(inputs)
		loss = criterion(outputs, targets)

		test_loss += loss.data
		_, predicted = outputs.max(1)
		total += targets.size(0)
		correct += predicted.eq(targets).sum().data

		progress_bar(batch_idx, len(valloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
			% (test_loss/(batch_idx+1), 100.*float(correct)/total, correct, total))

	# Save checkpoint.
	acc = (100.0*float(correct))/total
	if acc > best_acc:
		print('Saving..')
		state = {
			'net': net.state_dict(),
			'acc': acc,
			'epoch': epoch,
			'val_indices': val_indices
		}
		if not os.path.isdir('checkpoint'):
			os.mkdir('checkpoint')
		torch.save(state, './checkpoint/ckpt.t7')
		best_acc = acc

print('parameter count: ' + str( sum(p.numel() for p in net.parameters())))
for epoch in range(start_epoch, start_epoch+60):
	train(epoch)
	test(epoch)
	if epoch == 20:
		args.lr = .01
		for param_group in optimizer.param_groups:
			param_group['lr'] = args.lr
	if epoch == 40:
		args.lr = .001
