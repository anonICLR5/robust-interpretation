import torch
import explainers
from explainers import get_top_k_mask, top_k_intersection
import data_utils
import numpy as np
import scipy.stats
import pickle
import time
import glob
import os

def get_certified_top_k_intersect_for_one_sample(k, x, l2, stddev, eta, n_samples):
	flat = x.abs().reshape(-1).sort(descending=True)[0]
	sz = flat.size()[0]
	err = np.sqrt(np.log(2*sz/(1-eta))/(2*n_samples))
	up = flat[:k]
	down = (flat[k:2*k].flip(0)).cpu().numpy()
	upbound = scipy.stats.norm.cdf(scipy.stats.norm.ppf(up.cpu().numpy()-err)-2*l2/stddev)-err
	return (upbound > down).sum().item()



def experiment_corrected_variance():
	sz = 32*32*3
	k = int(.25*sz)
	model = data_utils.load_resnet_18_model()
	diffmodel = data_utils.load_soft_resnet_18_model()
	os.makedirs('./cifar_exp', exist_ok=True)
	most_base_explainer = explainers.BaseExplainerNoAbs(requires_grad=True)
	i = 0
	while(True):
		i = i + 1
		x = data_utils.load_cifar10_batch(batch_size=1)
		std_dev = .2*(x.max().item() - x.min().item())
		small_vanilla_smoothgrad_explainer = explainers.SmoothGradBatchFastExplainer(base_explainer=explainers.BaseExplainerNoAbs(requires_grad=True),
			stdev=std_dev, n_samples=64, batch_size = 64)
		large_vanilla_smoothgrad_explainer = explainers.SmoothGradBatchFastExplainer(base_explainer=explainers.BaseExplainerNoAbs(),
			stdev=std_dev, n_samples=8192, batch_size = 512)
		large_vanilla_smoothgrad_guider = explainers.SmoothGradBatchFastExplainer(base_explainer=explainers.BaseExplainerNoAbs(requires_grad=True),
			stdev=std_dev, n_samples=512, batch_size = 512)
		large_vanilla_squared_smoothgrad_explainer = explainers.SmoothGradBatchFastExplainer(base_explainer=explainers.BaseExplainerSquared(),
			stdev=std_dev, n_samples=8192, batch_size = 512)
		large_vanilla_squared_smoothgrad_guider = explainers.SmoothGradBatchFastExplainer(base_explainer=explainers.BaseExplainerSquared(requires_grad=True),
			stdev=std_dev, n_samples=512, batch_size = 512)
		large_robust_explainer = explainers.SmoothGradBatchFastExplainer(base_explainer=explainers.L0Explainer(int(.1*sz), flatten=int(.01*sz)),
			stdev=std_dev, n_samples=8192, batch_size = 512)
		large_robust_guider = explainers.SmoothGradBatchFastExplainer(base_explainer=explainers.L0Explainer(int(.1*sz), flatten=int(.01*sz), requires_grad=True),
			stdev=std_dev, n_samples=512, batch_size = 512)
		orig_base_grad = torch.abs(most_base_explainer.explain(model, x))
		orig_small_grad = torch.abs(small_vanilla_smoothgrad_explainer.explain(model, x))
		orig_large_smooth_grad = torch.abs(large_vanilla_smoothgrad_explainer.explain(model, x))
		orig_large_squared_smooth_grad = torch.abs(large_vanilla_squared_smoothgrad_explainer.explain(model, x))

		orig_large_robust_grad = large_robust_explainer.explain(model, x)
		mask_orig_base_grad = get_top_k_mask(orig_base_grad, k)
		mask_orig_small_grad = get_top_k_mask(orig_small_grad, k)
		mask_orig_large_smooth_grad = get_top_k_mask(orig_large_smooth_grad, k)
		mask_orig_large_robust_grad = get_top_k_mask(orig_large_robust_grad, k)
		mask_orig_large_squared_smooth_grad = get_top_k_mask(orig_large_squared_smooth_grad, k)
		for l2 in [.03, .1, 1, 10, 30]:
			cert = float(get_certified_top_k_intersect_for_one_sample(k, orig_large_robust_grad, l2, std_dev, .95, 8192))/k
			single_jump_attacker =  explainers.SingleJumpL2FastAttackerWithFailure(l2, 100)
			single_jump_attack, fail = single_jump_attacker.explain(model,x)
			if (not fail):
				single_jump_base_grad = torch.abs(most_base_explainer.explain(model, single_jump_attack))
				single_jump_small_grad = torch.abs(small_vanilla_smoothgrad_explainer.explain(model, single_jump_attack))
				single_jump_base_accuracy = top_k_intersection(mask_orig_base_grad, get_top_k_mask(single_jump_base_grad,k)).type(x.type())/k
				single_jump_small_accuracy = top_k_intersection(mask_orig_small_grad, get_top_k_mask(single_jump_small_grad,k)).type(x.type())/k
			list_dict = {
				'x' : x,
				'orig_base_grad': orig_base_grad,
				'orig_small_grad': orig_small_grad,
				'orig_large_smooth_grad': orig_large_smooth_grad,
				'orig_large_squared_smooth_grad': orig_large_squared_smooth_grad,
				'orig_large_robust_grad': orig_large_robust_grad,
				'l2': l2,
				'cert': cert
			}

			ghorbani_base_attack = explainers.EnhancedGhorbaniSingleSampleL2Attacker(l2, most_base_explainer, 100, 20, 5, k).explain(diffmodel, x, classifier_model = model)
			if (ghorbani_base_attack is not None):
				ghorbani_base_grad = torch.abs(most_base_explainer.explain(model, ghorbani_base_attack))
				ghorbani_base_accuracy = top_k_intersection(mask_orig_base_grad, get_top_k_mask(ghorbani_base_grad,k)).type(x.type())/k
				list_dict.update({
					'ghorbani_base_attack' :ghorbani_base_attack,
					'ghorbani_base_grad' :ghorbani_base_grad,
					'ghorbani_base_accuracy' :ghorbani_base_accuracy
				})
			pure_ghorbani_base_attack,pgfail = explainers.PureGhorbaniAttacker(l2, most_base_explainer, 100, l2/500, k).explain(diffmodel, x, classifier_model = model)
			if (not pgfail):
				pure_ghorbani_base_grad = torch.abs(most_base_explainer.explain(model, pure_ghorbani_base_attack))
				pure_ghorbani_base_accuracy = top_k_intersection(mask_orig_base_grad, get_top_k_mask(pure_ghorbani_base_grad,k)).type(x.type())/k
				list_dict.update({
					'pure_ghorbani_base_attack' :pure_ghorbani_base_attack,
					'pure_ghorbani_base_grad' :pure_ghorbani_base_grad,
					'pure_ghorbani_base_accuracy' :pure_ghorbani_base_accuracy
				})
			ghorbani_small_attack = explainers.EnhancedGhorbaniSingleSampleL2Attacker(l2, small_vanilla_smoothgrad_explainer, 100, 20, 5, k).explain(diffmodel, x, classifier_model = model)
			if (ghorbani_small_attack is not None):
				ghorbani_small_grad = torch.abs(small_vanilla_smoothgrad_explainer.explain(model, ghorbani_small_attack))
				ghorbani_small_accuracy = top_k_intersection(mask_orig_small_grad, get_top_k_mask(ghorbani_small_grad,k)).type(x.type())/k
				list_dict.update({
					'ghorbani_small_attack' :ghorbani_small_attack,
					'ghorbani_small_grad' :ghorbani_small_grad,
					'ghorbani_small_accuracy' :ghorbani_small_accuracy
				})
			pure_ghorbani_small_attack,pgfail = explainers.PureGhorbaniAttacker(l2, small_vanilla_smoothgrad_explainer, 100, l2/500, k).explain(diffmodel, x, classifier_model = model)
			if (not pgfail):
				pure_ghorbani_small_grad = torch.abs(small_vanilla_smoothgrad_explainer.explain(model, pure_ghorbani_small_attack))
				pure_ghorbani_small_accuracy = top_k_intersection(mask_orig_small_grad, get_top_k_mask(pure_ghorbani_small_grad,k)).type(x.type())/k
				list_dict.update({
					'pure_ghorbani_small_attack' :pure_ghorbani_small_attack,
					'pure_ghorbani_small_grad' :pure_ghorbani_small_grad,
					'pure_ghorbani_small_accuracy' :pure_ghorbani_small_accuracy
				})
			ghorbani_large_smoothgrad_attack = explainers.EnhancedGhorbaniSingleSampleL2Attacker(l2, large_vanilla_smoothgrad_guider, 100, 20, 5, k).explain(diffmodel, x, classifier_model = model)
			if (ghorbani_large_smoothgrad_attack is not None):
				ghorbani_large_smooth_grad = torch.abs(large_vanilla_smoothgrad_explainer.explain(model, ghorbani_large_smoothgrad_attack))
				ghorbani_large_smooth_accuracy = top_k_intersection(mask_orig_large_smooth_grad, get_top_k_mask(ghorbani_large_smooth_grad,k)).type(x.type())/k
				list_dict.update({
					'ghorbani_large_smoothgrad_attack' :ghorbani_large_smoothgrad_attack,
					'ghorbani_large_smooth_grad' :ghorbani_large_smooth_grad,
					'ghorbani_large_smooth_accuracy' :ghorbani_large_smooth_accuracy
				})
			ghorbani_large_squared_smoothgrad_attack = explainers.EnhancedGhorbaniSingleSampleL2Attacker(l2, large_vanilla_squared_smoothgrad_guider, 100, 20, 5, k).explain(diffmodel, x, classifier_model = model)
			if (ghorbani_large_squared_smoothgrad_attack is not None):
				ghorbani_large_squared_smooth_grad = torch.abs(large_vanilla_squared_smoothgrad_explainer.explain(model, ghorbani_large_squared_smoothgrad_attack))
				ghorbani_large_squared_smooth_accuracy = top_k_intersection(mask_orig_large_squared_smooth_grad, get_top_k_mask(ghorbani_large_squared_smooth_grad,k)).type(x.type())/k
				list_dict.update({
					'ghorbani_large_squared_smoothgrad_attack' :ghorbani_large_squared_smoothgrad_attack,
					'ghorbani_large_squared_smooth_grad' :ghorbani_large_squared_smooth_grad,
					'ghorbani_large_squared_smooth_accuracy' :ghorbani_large_squared_smooth_accuracy
				})
			ghorbani_large_robust_attack = explainers.EnhancedGhorbaniSingleSampleL2Attacker(l2, large_robust_guider, 100, 20, 5, k).explain(diffmodel, x, classifier_model = model)
			if (ghorbani_large_robust_attack is not None):
				ghorbani_large_robust_grad = large_robust_explainer.explain(model, ghorbani_large_robust_attack)
				ghorbani_large_robust_accuracy = top_k_intersection(mask_orig_large_robust_grad, get_top_k_mask(ghorbani_large_robust_grad,k)).type(x.type())/k
				list_dict.update({
					'ghorbani_large_robust_attack' :ghorbani_large_robust_attack,
					'ghorbani_large_robust_grad' :ghorbani_large_robust_grad,
					'ghorbani_large_robust_accuracy' : ghorbani_large_robust_accuracy
				})
			if (not fail):
				list_dict.update({
					'single_jump_attack' :single_jump_attack,
					'single_jump_base_grad' :single_jump_base_grad,
					'single_jump_small_grad':single_jump_small_grad,
					'single_jump_base_accuracy':single_jump_base_accuracy,
					'single_jump_small_accuracy':single_jump_small_accuracy
				})

			f1 = open('./cifar_exp/' + str(l2) + ' ' + str(i) + ' ' + str(time.time()) + '.pickle', 'wb')
			pickle.dump(list_dict, f1)
			f1.close()

experiment_corrected_variance()

