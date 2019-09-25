import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
import glob
import os
import json
from PIL import Image
import matplotlib.pyplot as plt
import cifar.models.cifar_resnet
import mnist.mnist_classifier
import cifar.models.softplus_cifar_resnet
import sys

def agg_default(x):
	if x.ndim == 4:
		return np.abs(x).sum(1)
	elif x.ndim == 3:
		return np.abs(x).sum(0)
epsilon = 1e-10
def clip(x, top_clip=True):
	if x.ndim == 3:
		batch_size, height, width = x.shape
		x = x.reshape(batch_size, -1)
		if top_clip:
			vmax = np.percentile(x, 99, axis=1, keepdims=True)
		else:
			vmax = np.max(x, axis=1, keepdims=True)
		vmin = np.min(x, axis=1, keepdims=True)
		vdiff = vmax - vmin
		for i, v in enumerate(vdiff):
			v = max(0, np.abs(v))
			if np.abs(v) < epsilon:
				x[i] = np.zeros_like(x[i])
			else:
				x[i] = np.clip((x[i] - vmin[i]) / v, 0, 1)
		x = x.reshape(batch_size, height, width)
	elif x.ndim == 2:
		height, width = x.shape
		x = x.ravel()
		x = np.nan_to_num(x)
		vmax = np.percentile(x, 99) if top_clip else np.max(x)
		vmin = np.min(x)
		vdiff = max(0, np.abs(vmax - vmin))
		if np.abs(vdiff) < epsilon:
			x = np.zeros_like(x)
		else:
			x = np.clip((x - vmin) / (vmax - vmin), 0, 1)
		x = x.reshape(height, width)
	return x

def agg_clip(x, top_clip=True):
	return clip(agg_default(x), top_clip=top_clip)

def plot_saliency(mp, file):
	result1 = np.uint8(255*agg_clip(mp.cpu().numpy()))
	plt.figure()
	plt.imshow(result1, cmap='gray')
	plt.savefig(file)

def load_resnet_18_model():
	model =  cifar.models.cifar_resnet.cifar_ResNet18()
	model = torch.nn.DataParallel(model).cuda()
	checkpoint = torch.load('./cifar/checkpoint/ckptval.t7')
	torch.backends.cudnn.enabled = False
	model.load_state_dict(checkpoint['net'])
	model.cuda()
	model.eval()
	return model

def load_mnist_model():
	model =  mnist.mnist_classifier.MnistNet()
	model = torch.nn.DataParallel(model).cuda()
	checkpoint = torch.load('./mnist/checkpoint/ckpt.t7')
	torch.backends.cudnn.enabled = False
	model.load_state_dict(checkpoint['net'])
	model.cuda()
	model.eval()
	return model

def load_soft_resnet_18_model():
	model = cifar.models.softplus_cifar_resnet.soft_cifar_ResNet18()
	model = torch.nn.DataParallel(model).cuda()
	checkpoint = torch.load('./cifar/soft_checkpoint/ckptval.t7')
	torch.backends.cudnn.enabled = False
	model.load_state_dict(checkpoint['net'])
	model.cuda()
	model.eval()
	return model
transform_test = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

mnist_transform = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize((0.1307,), (0.3081,)),
])

def load_cifar10_batch(batch_size = 64):
	testset = torchvision.datasets.CIFAR10(root='./cifar/data', train=False, download=True, transform=transform_test)
	testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=2)
	(inputs, targets) = next(iter(testloader))
	return inputs.cuda()

def load_mnist_batch(batch_size = 256):
	testset = torchvision.datasets.MNIST(root='./mnist/data', train=False, download=True, transform=mnist_transform)
	testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=2)
	(inputs, targets) = next(iter(testloader))
	return inputs.cuda()

def pil_loader(path):
	with open(path, 'rb') as f:
		with Image.open(f) as img:
			return img.convert('RGB')

def setup_imagenet(batch_size=16, example_ids=None,
				   n_batches=-1, n_examples=-1,
				   shuffle=True, dump_name=None):
	model = torchvision.models.resnet50(pretrained=True)
	model.eval()
	model.cuda()
	print('model loaded')

	home_dir = './ILSVRC_val/'
	image_path = './ILSVRC_val/**/*.JPEG'
	image_files = list(glob.iglob(image_path, recursive=True))
#    print(len(image_files))
	image_files = sorted(image_files, key=lambda x: os.path.basename(x))
	real_ids = [os.path.basename(x) for x in image_files]

	if example_ids is not None:
		examples = {r: (r, m)
					for r, m in zip(real_ids, image_files)}
		examples = [examples[x] for x in example_ids]
	else:
		examples = list(zip(real_ids, image_files))

	if shuffle:
		np.random.seed(0)
		np.random.shuffle(examples)

	if n_examples > 0:
		examples = examples[:n_examples]
	elif n_batches > 0:
		examples = examples[:batch_size * n_batches]
	else:
		print('using all images')

	selected_files = sorted([x[0] for x in examples])
	if dump_name is not None:
		with open(dump_name, 'w') as f:
			f.write(json.dumps(selected_files))
#    print('\n'.join(selected_files))

	def batch_loader(batch):
		batch = list(map(list, zip(*batch)))
		ids, xs = batch
		return (ids, [pil_loader(x) for x in xs])

	batch_indices = list(range(0, len(examples), batch_size))
	batches = [examples[i: i + batch_size] for i in batch_indices]
	batches = map(batch_loader, batches)
#    print('image loaded', len(batch_indices))
	return model, batches
