import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import math
import os
import explainers
explainers.mnist_scale =True
import torchviz
import os
import argparse

from mnist.mnist_classifier import MnistNet
import data_utils

import sys
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.gridspec as gridspec

from explainers import get_top_k_mask, top_k_intersection


inv_normalize = transforms.Normalize(
	mean=[-0.1307/0.3081,],
	std=[1/0.3081,]
)

def plot_images(orig, orig_saliencies, attacks, attack_saliencies, file):

	f, axarr = plt.subplots(orig.shape[0] ,4,gridspec_kw={'wspace':0, 'hspace':0})
	f.set_figheight(orig.shape[0])
	f.set_figwidth(4)
	for i in range(orig.shape[0]):

		axarr[i,0].imshow(inv_normalize(orig[i].cpu()).numpy()[0])
		axarr[i,0].axis('off')
		axarr[i,2].imshow(inv_normalize(attacks[i].cpu()).numpy()[0])
		axarr[i,2].axis('off')
		result1 = np.uint8(255*data_utils.agg_clip(orig_saliencies[i].cpu().detach().numpy()))
		axarr[i,1].imshow(result1.reshape(28,28), cmap='gray')
		axarr[i,1].axis('off')
		result1 = np.uint8(255*data_utils.agg_clip(attack_saliencies[i].cpu().detach().numpy()))
		axarr[i,3].imshow(result1.reshape(28,28), cmap='gray')
		axarr[i,3].axis('off')
	plt.savefig(file)

def plot_images_comparison(orig, orig_saliencies, attacks, attack_saliencies, lambdas, file):
	cols = 1+4*len(orig_saliencies)
	f, axarr = plt.subplots(orig.shape[0] ,cols,gridspec_kw={'wspace':0, 'hspace':0, 'width_ratios': [1,.5,1,1,1,.5,1,1,1,.5,1,1,1]})
	f.set_figheight(orig.shape[0]+.3)
	f.set_figwidth(cols*.7)
	for i in range(orig.shape[0]):
		if (i ==0):
			axarr[i,0].set_title('')
		axarr[i,0].imshow(inv_normalize(orig[i].cpu()).numpy()[0])
		axarr[i,0].axis('off')
		for j in range(len(orig_saliencies)):
			if (i ==0):
				axarr[i,4*j+3].set_title('λ͟=͟'+lambdas[j]+'\n')
				axarr[i,4*j+2].set_title('')
				axarr[i,4*j+4].set_title('')
			axarr[i,4*j+3].imshow(inv_normalize(attacks[j][i].cpu()).numpy()[0])
			axarr[i,4*j+3].axis('off')
			result1 = np.uint8(255*data_utils.agg_clip(orig_saliencies[j][i].cpu().detach().numpy()))
			axarr[i,4*j+2].imshow(result1.reshape(28,28), cmap='gray')
			axarr[i,4*j+2].axis('off')
			result1 = np.uint8(255*data_utils.agg_clip(attack_saliencies[j][i].cpu().detach().numpy()))
			axarr[i,4*j+4].imshow(result1.reshape(28,28), cmap='gray')
			axarr[i,4*j+4].axis('off')
			axarr[i,4*j+1].axis('off')
	plt.savefig(file)

parser = argparse.ArgumentParser(description='PyTorch MNIST Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')

transform = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize((0.1307,), (0.3081,)),
])



cudnn.benchmark = True

trainset = torchvision.datasets.MNIST(root='./mnist/data', train=True, download=True, transform=transform)
testset = torchvision.datasets.MNIST(root='./mnist/data', train=False, download=True, transform=transform)



checkpoint = torch.load('./mnist/checkpoint/ckpt.t7')
val_indices = checkpoint['val_indices']



train_indices = list(set(range(len(trainset))) - set(val_indices))
trainloader = torch.utils.data.DataLoader(trainset, batch_size=512, num_workers=2, sampler=torch.utils.data.sampler.SubsetRandomSampler(train_indices))
valloader = torch.utils.data.DataLoader(trainset, batch_size=512, num_workers=2, sampler=torch.utils.data.sampler.SubsetRandomSampler(val_indices))

def test_attacks(c,model, explainer, attacker, loss_ratio):
	sz = 28*28
	k = int(.25*sz)
	(batch, ground_truths) = next(iter(valloader))
	images = batch.cuda()
	attacks,fail = attacker.explain(model, images.clone())
	#print('fails: ' + str(fail.sum()))
	images = images[(~fail).nonzero()].squeeze(dim=1)
	attacks = attacks[(~fail).nonzero()].squeeze(dim=1)
	base_grad = explainer.explain(model, images)
	base_mask = get_top_k_mask(base_grad, k)
	attacked_grad = explainer.explain(model,attacks)
	attacked_mask = get_top_k_mask(attacked_grad,k)
	accuracies = top_k_intersection(base_mask, attacked_mask).type(images.type())/k
	if (c == 29):
		plot_images(images[:16],base_grad[:16],attacks[:16],attacked_grad[:16],'./mnist_robust_checkpoint/example_attacks_' + str(loss_ratio)+'_'+ str(accuracies.mean())+'.png')
	return accuracies.mean()

def normalize_each(x):
	flattened = x.reshape(x.shape[0],-1)
	return ((flattened.t()/flattened.norm(dim=1)).t()).reshape(x.shape)

def standardplot( attacker, explainer):
	criterion = nn.CrossEntropyLoss()
	sz = 28*28
	k = int(.25*sz)
	frs = []
	accs = []
	salax = []
	testloader = torch.utils.data.DataLoader(testset, batch_size=3, shuffle=True, num_workers=2)
	(inputs, targets) =next(iter(testloader))
	inputs, targets = Variable(inputs).cuda(), Variable(targets).cuda()
	basegrads = []
	attackss = []
	attackedgrads = []
	for lmbda in [0., 200., 1000.]:
		model =  MnistNet()
		model = torch.nn.DataParallel(model).cuda()
		checkpoint = torch.load('./mnist_robust_checkpoint/ckpt'+str(lmbda)+ '_29.t7')
		torch.backends.cudnn.enabled = False
		model.load_state_dict(checkpoint['net'])
		model.cuda()
		model.eval()
		correct = 0
		total = 0
		fails = 0
		accuracieslist = []
		outputs = model(inputs)
		loss = criterion(outputs, targets)
		_, predicted = outputs.max(1)
		total += targets.size(0)
		correct += predicted.eq(targets).sum().data
		attacks,fail = attacker.explain(model, inputs.clone())
		fails += fail.sum()
		images = inputs[(~fail).nonzero()].squeeze(dim=1)
		attacks = attacks[(~fail).nonzero()].squeeze(dim=1)
		base_grad = explainer.explain(model, images)
		attacked_grad = explainer.explain(model,attacks)
		allimages = images[:3]

		basegrads.append(base_grad[:3])
		attackss.append(attacks[:3])
		attackedgrads.append(attacked_grad[:3]),
	plot_images_comparison(allimages,basegrads,attackss,attackedgrads,['0͟', '2͟0͟0͟','1͟0͟0͟0͟'],'./mnist_robust_checkpoint/test_attacks_samecols2.png')



def test( attacker, explainer):
	criterion = nn.CrossEntropyLoss()
	sz = 28*28
	k = int(.25*sz)
	frs = []
	accs = []
	salax = []
	for lmbda in [0., 200., 400., 600., 800., 1000.]:
		model =  MnistNet()
		model = torch.nn.DataParallel(model).cuda()
		checkpoint = torch.load('./mnist_robust_checkpoint/ckpt'+str(lmbda)+ '_29.t7')
		torch.backends.cudnn.enabled = False
		model.load_state_dict(checkpoint['net'])
		model.cuda()
		model.eval()
		correct = 0
		total = 0
		fails = 0
		accuracieslist = []
		testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=True, num_workers=2)
		for batch_idx, (inputs, targets) in enumerate(testloader):
			inputs, targets = Variable(inputs).cuda(), Variable(targets).cuda()
			outputs = model(inputs)
			loss = criterion(outputs, targets)
			_, predicted = outputs.max(1)
			total += targets.size(0)
			correct += predicted.eq(targets).sum().data
			attacks,fail = attacker.explain(model, inputs.clone())
			fails += fail.sum()
			images = inputs[(~fail).nonzero()].squeeze(dim=1)
			attacks = attacks[(~fail).nonzero()].squeeze(dim=1)
			base_grad = explainer.explain(model, images)
			base_mask = get_top_k_mask(base_grad, k)
			attacked_grad = explainer.explain(model,attacks)
			attacked_mask = get_top_k_mask(attacked_grad,k)
			accuracies = top_k_intersection(base_mask, attacked_mask).type(images.type())/k
			accuracieslist.append(accuracies)
			if (batch_idx == 0):
				plot_images(images[:3],base_grad[:3],attacks[:3],attacked_grad[:3],'./mnist_robust_checkpoint/test_attacks_' + str(lmbda)+'.png')
	
		failrate = (100.0*float(fails))/total
		acc = (100.0*float(correct))/total
		salacc = torch.cat(accuracieslist).mean()*100
		print(failrate)
		print(acc)
		print(salacc)
		frs.append(failrate)
		accs.append(acc)
		salax.append(salacc)
	record = {
		'fails': frs,
		'accs' : accs,
		'salax': salax,
		'lmd':  [0, 200, 400, 600, 800, 1000]
		}
	torch.save(record, './mnist_robust_checkpoint/logstest.t7')

def plotfinal():
	rec = torch.load('./mnist_robust_checkpoint/logstest.t7')

	fig, ax1 = plt.subplots()

	color = 'tab:red'
	ax1.set_xlabel('λ', fontsize=14)
	ax1.set_ylabel('Robustness', color=color, fontsize=14)
	ax1.plot(rec['lmd'], [float(x)/100. for x in rec['salax']], color=color)
	ax1.tick_params(axis='y', labelcolor=color)
	ax1.set_ylim(bottom=.5, top=1)
	ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

	color = 'tab:blue'
	ax2.set_ylabel('Accuracy', color=color, fontsize=14)  # we already handled the x-label with ax1
	ax2.plot(rec['lmd'], [float(x)/100. for x in rec['accs']], color=color)
	ax2.tick_params(axis='y', labelcolor=color)
	ax2.set_ylim(bottom=.5, top=1)
	fig.tight_layout()  # otherwise the right y-label is slightly clipped
	plt.savefig('./mnist_robust_checkpoint/outfinal.png')
def plot():
	rec = torch.load('./mnist_robust_checkpoint/logstest.t7')
	plt.plot(rec['lmd'],rec['accs'], label='Classification accuracy')
	plt.plot(rec['lmd'],rec['salax'], label='Top-25% overlap from unperturbed saliency map')
	plt.xlabel('λ')
	plt.ylabel('%')
	plt.legend()
	plt.title('Effectiveness of adversarial training')
	plt.savefig('./mnist_robust_checkpoint/out.png')
def test_adv_class( attacker, explainer):
	criterion = nn.CrossEntropyLoss()
	sz = 28*28
	k = int(.25*sz)
	frs = []
	accs = []
	salax = []
	bmodel =  MnistNet().cuda()
	#model = torch.nn.DataParallel(model).cuda()
	checkpoint = torch.load('./mnist_robust_checkpoint/mnist_adv_2.4.pth')
	torch.backends.cudnn.enabled = False
	bmodel.load_state_dict(checkpoint)
	bmodel.cuda()
	bmodel.eval()
	model = lambda x: bmodel(x*0.3081 + 0.1307)
	correct = 0
	total = 0
	fails = 0
	accuracieslist = []
	testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=True, num_workers=2)
	for batch_idx, (inputs, targets) in enumerate(testloader):
		inputs, targets = Variable(inputs).cuda(), Variable(targets).cuda()
		outputs = model(inputs)
		loss = criterion(outputs, targets)
		_, predicted = outputs.max(1)
		total += targets.size(0)
		correct += predicted.eq(targets).sum().data
		attacks,fail = attacker.explain(model, inputs.clone())
		fails += fail.sum()
		images = inputs[(~fail).nonzero()].squeeze(dim=1)
		attacks = attacks[(~fail).nonzero()].squeeze(dim=1)
		base_grad = explainer.explain(model, images)
		base_mask = get_top_k_mask(base_grad, k)
		attacked_grad = explainer.explain(model,attacks)
		attacked_mask = get_top_k_mask(attacked_grad,k)
		accuracies = top_k_intersection(base_mask, attacked_mask).type(images.type())/k
		accuracieslist.append(accuracies)
		if (batch_idx == 0):
			plot_images(images[:3],base_grad[:3],attacks[:3],attacked_grad[:3],'./mnist_robust_checkpoint/test_attacks_advtrained.png')
		print('batch')
	failrate = (100.0*float(fails))/total
	acc = (100.0*float(correct))/total
	salacc = torch.cat(accuracieslist).mean()*100
	print(failrate)
	print(acc)
	print(salacc)
	frs.append(failrate)
	accs.append(acc)
	salax.append(salacc)
	record = {
		'fails': frs,
		'accs' : accs,
		'salax': salax,
		}
	torch.save(record, './mnist_robust_checkpoint/advtrained.t7')

def train(num_cycles, attacker, explainer, loss_ratio):
	model =  MnistNet()
	model = torch.nn.DataParallel(model).cuda()
	best_loss=None

	print('lambda: ' + str(loss_ratio))
	model.train()
	criterionCEL = nn.CrossEntropyLoss()
	criterionMSE = nn.MSELoss()
	optimizer = optim.Adam(model.parameters())
	worst_loss = math.inf
	#if(best_loss):
	#	worst_loss=best_loss
	last_val_save = 0
	best_correct = None
	best_sal_accuracy = None
	for c in range(num_cycles):
		(batch, ground_truths) = next(iter(valloader))
		batch=batch.cuda()
		var_x = Variable(batch, requires_grad = True)
		base_saliencies = normalize_each(torch.abs(explainer.explain(model, var_x)))
		all_attacks = []
		all_attack_saliencies = []
		all_attack_labels = []
		all_attack_baseline_saliencies = []

		attacks,fail = attacker.explain(model, batch)
		succeed_indices = (~fail).nonzero()
		attacks= attacks[succeed_indices].squeeze(dim=1)
		all_attacks.append(attacks)
		all_attack_saliencies.append(normalize_each(torch.abs(explainer.explain(model, attacks))))
		all_attack_labels.append(ground_truths[succeed_indices].squeeze(dim=1))
		all_attack_baseline_saliencies.append(base_saliencies[succeed_indices].squeeze(dim=1))

		all_classifications = torch.cat([torch.cat(all_attacks),var_x])
		#all_labels = torch.cat([torch.cat(all_attack_labels),ground_truths]).cuda()
		all_labels = ground_truths.cuda()
		all_attack_saliencies = torch.cat(all_attack_saliencies)
		all_attack_baseline_saliencies = torch.cat(all_attack_baseline_saliencies)
		total = all_labels.size(0)
		#predicted = model(all_classifications)
		predicted = model(var_x)
		correct = 100.*float(predicted.max(1)[1].eq(all_labels).sum().data)/total
		mseloss = loss_ratio*criterionMSE(all_attack_saliencies, all_attack_baseline_saliencies)
		loss = criterionCEL(predicted, all_labels) + mseloss
		saliency_error = test_attacks(c,model, explainer, attacker, loss_ratio)

		#if (loss < worst_loss or (c+ start_epoch)==1):
		#worst_loss = loss
		print('Saving..')
		print('iteration: '+str(c+ start_epoch))
		print('test loss: ' + str(loss))
		print('test accuracy: ' + str(correct))
		print('test MSE: '  + str(mseloss))
		print('test saliency error: '  + str(saliency_error))
		last_val_save = c
		best_correct = correct
		best_sal_accuracy = saliency_error
		state = {
			'net': model.state_dict(),
			'loss': loss,
			'test_acc': correct,
			'test_sal_err': saliency_error,
			'epoch': c+start_epoch,
			'val_indices': val_indices
		}
		if not os.path.isdir('mnist_robust_checkpoint'):
			os.mkdir('mnist_robust_checkpoint')
		torch.save(state, './mnist_robust_checkpoint/ckpt' +str(loss_ratio)+'_'+str(c+start_epoch)+'.t7')


		for batch_idx, (inputs, ground_truths) in enumerate(trainloader):
			optimizer.zero_grad()
			batch = inputs.cuda()
			var_x = Variable(batch, requires_grad = True)
			
			#print(torchviz.make_dot(base_saliencies))
			all_attack_saliencies = []
			all_attack_labels = []
			all_attack_baseline_saliencies = []
			
			attacks,fail = attacker.explain(model, batch)

			succeed_indices = (~fail).nonzero()
			attacks= attacks[succeed_indices].squeeze(dim=1)
			num_succeed = attacks.shape[0]
			orig = var_x[succeed_indices].squeeze(dim=1)

			v = Variable(attacks,requires_grad=True)
			packed = torch.cat([v,orig])
			logits = model(packed)
			max_logit, y = logits.max(1)
			a, = torch.autograd.grad(max_logit.sum(), packed, create_graph=True)
	
			all_attack_saliencies = normalize_each(torch.abs(a[0:num_succeed]))
			all_attack_labels =ground_truths[succeed_indices].squeeze(dim=1)
			all_attack_baseline_saliencies = normalize_each(torch.abs(a[num_succeed:]))
			all_labels = ground_truths.cuda()
			mseloss = loss_ratio*criterionMSE(all_attack_saliencies, all_attack_baseline_saliencies)
			loss = mseloss+ criterionCEL(model(var_x), all_labels)
			loss.backward()
			optimizer.step()
	return best_correct,best_sal_accuracy
explainer = explainers.BaseExplainerNoAbs(requires_grad=True)
attacker = explainers.EnhancedGhorbaniBatchL2Attacker(10, explainer, 100, 15, 3, 7*28)
corrs = []
accs = []
lambs = []
for i in [0, 200, 400, 600, 800, 1000 ]:
	corr, acc = train(30,attacker, explainer, 1.0*i)
	corrs.append(corr)
	accs.append(acc)
	lambs.append(1.0*i)
	record = {
		'corrects': corrs,
		'accs' : accs,
		'lambs': lambs
	}
	torch.save(record, './mnist_robust_checkpoint/logs3.t7')
test(attacker,explainer)
standardplot(attacker,explainer)
plotfinal()
#test_adv_class( attacker, explainer)