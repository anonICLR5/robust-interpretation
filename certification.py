import explainers
explainers.imagenet_scale =True
import numpy as np
import pickle
import sys
import torchvision.transforms as transforms
import torch
import scipy.stats
import time
import matplotlib
import matplotlib.pyplot as plt
import data_utils
transf = transforms.Compose([
		transforms.Resize(256),
		transforms.CenterCrop(224),
		transforms.ToTensor()
	])

def load_gradients(density, stddev, path):
	return pickle.load(open(str(density) + '_' + str(stddev) + path, 'rb'))
def percentile(scores, percent):
	return scores.sort()[0][int(np.ceil((scores.size()[0]-1)*(percent)))]
def get_certified_top_k_intersect_for_one_sample(k, x, l2, stddev, eta, n_samples):
	flat = x.abs().reshape(-1).sort(descending=True)[0]
	sz = flat.size()[0]
	err = np.sqrt(np.log(2*sz/(1-eta))/(2*n_samples))
	up = flat[:k]
	down = (flat[k:2*k].flip(0)).cpu().numpy()
	upbound = scipy.stats.norm.cdf(scipy.stats.norm.ppf(up.cpu().numpy()-err)-2*l2/stddev)-err
	return (upbound > down).sum().item()
def get_certified_top_k_intersect_percentile_per_sample(k, x, l2, stddev, eta, n_samples):
	rescount = torch.tensor([get_certified_top_k_intersect_for_one_sample(k, v, l2, stddev, eta, n_samples) for v in x])
	#print(rescount.float()/float(k))
	return rescount.float()/float(k)
def plot_topk_bounds_final(k_frac, attack_magnitude, path, data_label):
	#plt.figure(figsize=(10,10))
	colors = ['g', 'r', 'c', 'blue', 'purple']
	for (idx, stddev) in enumerate([.02, .05, .1, .2, .5]):
		bounds = []
		uperror = []
		lowerror = []
		for density in [.02, .05, .1 ,.2, .5]:
			gradients = load_gradients(density, stddev, path)
			sz = gradients[0].nelement()
			k = int(k_frac*sz)
			bound = percentile(get_certified_top_k_intersect_percentile_per_sample(k, gradients, attack_magnitude, stddev, .95, 8192), .6)
			lo = percentile(get_certified_top_k_intersect_percentile_per_sample(k, gradients, attack_magnitude, stddev, .95, 8192), .72)
			up = percentile(get_certified_top_k_intersect_percentile_per_sample(k, gradients, attack_magnitude, stddev, .95, 8192), .48)
			bounds.append(bound)
			uperror.append(up-bound)
			lowerror.append(bound-lo)
		plt.errorbar([.02, .05, .1 ,.2, .5], bounds, yerr = [lowerror, uperror], label='σ = ' + str(stddev), color=colors[idx])
	plt.gca().set_xscale('log')
	plt.gca().set_xticks([.02, .05, .1 ,.2, .5])
	plt.gca().get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
	plt.gca().set_ylim(bottom=-.05, top=.65)
	plt.xlabel('Sparsification Parameter', fontsize=14)
	plt.ylabel('Robustness Certificate', fontsize=14)
	plt.legend()
	#plt.title('Certified Top '+ str(100.0*k_frac) + '% overlap after L₂= ' + str(attack_magnitude) + ' Attack, using ' + data_label +'.')
	plt.savefig('Certified Top '+ str(100.0*k_frac) + '% overlap after L2= ' + str(attack_magnitude) + ' Attack, using ' + data_label +'.png')

def test_zero_one_no_interpretation(model, x, n_samples=8192):
	sz = x[0].nelement()
	for density in [.02, .05, .1 ,.2, .5]:
		for stddev in [.02, .05, .1, .2, .5]:
			l0 = int(density*sz)
			explainer = explainers.SmoothGradRobustExplainer(base_explainer=explainers.ZeroOneExplainer(l0),
				stdev=stddev, image_size=sz, n_samples=n_samples)
			gradients = explainer.explain(model,x)
			f1 = open(str(density) + '_' + str(stddev) + '_unprocessed.pickle', 'wb')
			pickle.dump(gradients, f1)
def test_just_l0_no_interpretation(model, x, n_samples=8192):
	sz = x[0].nelement()
	for density in [.02, .05, .1 ,.2, .5, 1]:
		for stddev in [.02, .05, .1, .2, .5]:
			l0 = int(density*sz)
			explainer = explainers.SmoothGradRobustExplainer(base_explainer=explainers.L0Explainer(l0),
				stdev=stddev, image_size=sz, n_samples=n_samples)
			gradients = explainer.explain(model,x)
			f1 = open(str(density) + '_' + str(stddev) + '_score_unprocessed.pickle', 'wb')
			pickle.dump(gradients, f1)

def test_l1_l0_no_interpretation(model, x, n_samples=8192):
	sz = x[0].nelement()
	for density in [.02, .05, .1 ,.2, .5, 1]:
		for stddev in [.02, .05, .1, .2, .5]:
			l0 = int(density*sz)
			l1 = int(density*sz*.5)
			explainer = explainers.SmoothGradRobustExplainer(base_explainer=explainers.L0L1Explainer(l0, l1),
				stdev=stddev, image_size=sz, n_samples=n_samples)
			gradients = explainer.explain(model,x)
			f1 = open(str(density) + '_' + str(stddev) + '_score_l1_l0_unprocessed.pickle', 'wb')
			pickle.dump(gradients, f1)
def test_just_linear_no_interpretation(model, x, n_samples=8192):
	sz = x[0].nelement()
	for density in [.02, .05, .1 ,.2, .5, 1]:
		for stddev in [.02, .05, .1, .2, .5]:
			l0 = int(density*sz)
			explainer = explainers.SmoothGradRobustExplainer(base_explainer=explainers.LinearExplainer(l0),
				stdev=stddev, image_size=sz, n_samples=n_samples)
			gradients = explainer.explain(model,x)
			f1 = open(str(density) + '_' + str(stddev) + '_score_linear_rank_unprocessed.pickle', 'wb')
			pickle.dump(gradients, f1)

def test_just_l0_cutoff_no_interpretation(model, x, n_samples=8192):
	sz = x[0].nelement()
	for density in [.02, .05, .1 ,.2, .5, 1]:
		for stddev in [.02, .05, .1, .2, .5]:
			l0 = int(density*sz)
			explainer = explainers.SmoothGradRobustExplainer(base_explainer=explainers.L0Explainer(l0, flatten=int(.01*sz)),
				stdev=stddev, image_size=sz, n_samples=n_samples)
			gradients = explainer.explain(model,x)
			f1 = open(str(density) + '_' + str(stddev) + '_score_.01_cutoff_unprocessed.pickle', 'wb')
			pickle.dump(gradients, f1)
def test_squared_no_interpretation(model, x, n_samples=8192):
	sz = x[0].nelement()
	for stddev in [.02, .05, .1, .2, .5]:
		explainer = explainers.SmoothGradRobustExplainer(base_explainer=explainers.ScaledExplainerSquared(),
			stdev=stddev, image_size=sz, n_samples=n_samples)
		gradients = explainer.explain(model,x)
		f1 = open(str(stddev) + '_score_squared_unprocessed.pickle', 'wb')
		pickle.dump(gradients, f1)
def test_noabs_no_interpretation(model, x, n_samples=8192):
	sz = x[0].nelement()
	for stddev in [.02, .05, .1, .2, .5]:
		explainer = explainers.SmoothGradRobustExplainer(base_explainer=explainers.ScaledExplainerNoAbs(),
			stdev=stddev, image_size=sz, n_samples=n_samples)
		gradients = explainer.explain(model,x)
		f1 = open(str(stddev) + '_score_noabs_unprocessed.pickle', 'wb')
		pickle.dump(gradients, f1)
# Example use:
model, batches = data_utils.setup_imagenet(batch_size=64, n_batches=1)
batches = list(batches)
batch = batches[0]
xs = torch.stack([transf(x) for x in batch[1]]).cuda()
test_zero_one_no_interpretation(model,xs)
plot_topk_bounds_final(.2,.03,'_unprocessed.pickle', 'Sparsified SmoothGrad')