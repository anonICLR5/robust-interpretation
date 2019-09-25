import sys
import numpy as np
import torch
import scipy.optimize
from torch.autograd import Variable
import torchviz
import math

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

mean_tensor = torch.cuda.FloatTensor(mean).view(1, -1, 1, 1)
std_tensor = torch.cuda.FloatTensor(std).view(1, -1, 1, 1)

# default behavior is for cifar10
imagenet_scale= False
mnist_scale=False

def clamp_batch(x):
	if (imagenet_scale):
		pass
	elif (mnist_scale):
		x.clamp_(min=-.4242, max= 2.821)
	else:
		x[:,0] = torch.clamp(x[:,0], min=-2.429, max= 2.514)
		x[:,1] = torch.clamp(x[:,1], min=-2.418, max= 2.596)
		x[:,2] = torch.clamp(x[:,2], min=-2.221, max= 2.753)


def normalize_if_necessary(x, x_0, l2):
	x_flat = x.reshape(x.shape[0],-1)
	x_0_flat = x_0.reshape(x.shape[0],-1)
	diff_flat = x_flat - x_0_flat
	norms = diff_flat.norm(dim=1)
	x_flat[(norms > l2).nonzero()] =  (x_0_flat + (l2*diff_flat.t()/norms).t())[(norms > l2).nonzero()]
	clamp_batch(x)

def forward(model, x):
	if (imagenet_scale):
		x_norm = (x - mean_tensor)/std_tensor
	else:
		x_norm = x
	logits = model(x_norm)
	return logits

def get_top_k_mask(x, k):
	flattened = x.reshape(x.shape[0], -1)
	x_sparsified = (flattened.scatter(
		1,
		torch.topk(flattened,x[0].nelement()-k,largest=False,dim=1)[1],
		0).reshape(x.shape) > 0).type(x.type())
	return x_sparsified
def top_k_intersection(x_1_mask, x_2_mask):
	x_l_flat = x_1_mask.reshape(x_1_mask.shape[0], -1)
	x_2_flat = x_2_mask.reshape(x_2_mask.shape[0], -1)
	return (x_l_flat*x_2_flat).sum(dim=1)
class Explainer: 
	def explain(self, model, x):
		pass
class Attacker: 
	def explain(self, model, x):
		pass
class SmoothGradRobustExplainer(Explainer):
	'''
	See https://arxiv.org/abs/1706.03825.
	'''
	def __init__(self, base_explainer=None, stdev=0.15, image_size=None, n_samples=16):
		self.base_explainer = base_explainer
		self.stdev = stdev
		self.n_samples = n_samples
		self.image_size = image_size

	def explain(self, model, x):
		total_gradients = 0
		for i in range(self.n_samples):
			noise = torch.randn(x.shape).cuda() * self.stdev
			x_var = noise + x.clone()
			grad = self.base_explainer.explain(model, x_var)
			total_gradients += grad
		total_gradients /= self.n_samples
		return total_gradients

class L0Explainer(Explainer):

	def __init__(self, l_0, flatten = 0, requires_grad=False):
		self.l_0 = l_0
		self.flatten = flatten
		self.requires_grad = requires_grad

	def explain(self, model, x):
		if (not x.requires_grad):
			x = Variable(x, requires_grad=True)
		logits = forward(model, x)
		max_logit, y = logits.max(1)
		x_grad, = torch.autograd.grad(max_logit.sum(), x, create_graph=self.requires_grad)
		x_grad = torch.abs(x_grad)
		if (self.flatten != 0):
			maxes = torch.topk(x_grad.reshape(x_grad.shape[0],-1),self.flatten,dim=1)[0].min(dim=1)[0]
			x_grad = (torch.where(x_grad.reshape(x_grad.shape[0],-1).t() < maxes, x_grad.reshape(x_grad.shape[0],-1).t(), maxes)/maxes).t().reshape(x_grad.shape)
		else:
			x_grad = (x_grad.reshape(x_grad.shape[0],-1).t()/x_grad.reshape(x_grad.shape[0],-1).max(dim=1)[0]).t().reshape(x_grad.shape)
		x_sparsified = x_grad.reshape(
			x_grad.shape[0],-1).scatter(
				1,
				torch.topk(x_grad.reshape(x_grad.shape[0],-1),x_grad[0].nelement()-self.l_0,largest=False,dim=1)[1],
				0).reshape(x_grad.shape)
		return x_sparsified
class L0AloneExplainer(Explainer):

	def __init__(self, l_0):
		self.l_0 = l_0
	def explain(self, model, x):
		x = Variable(x, requires_grad=True)
		logits = forward(model, x)
		max_logit, y = logits.max(1)
		x_grad, = torch.autograd.grad(max_logit.sum(), x)
		x_grad = torch.abs(x_grad)
		x_sparsified = x_grad.reshape(
			x_grad.shape[0],-1).scatter(
				1,
				torch.topk(x_grad.reshape(x_grad.shape[0],-1),x_grad[0].nelement()-self.l_0,largest=False,dim=1)[1],
				0).reshape(x_grad.shape)
		return x_sparsified
class BaseExplainer(Explainer):
	def __init__(self, requires_grad=False):
		self.requires_grad = requires_grad
	def explain(self, model, x):
		if (not x.requires_grad):
			x = Variable(x, requires_grad=True)
		logits = forward(model, x)
		max_logit, y = logits.max(1)
		x_grad, = torch.autograd.grad(max_logit.sum(), x, create_graph=self.requires_grad)
		x_grad = torch.abs(x_grad)
		return x_grad

class BaseExplainerNoAbs(Explainer):
	def __init__(self, requires_grad=False):
		self.requires_grad = requires_grad
	def explain(self, model, x):
		if (not x.requires_grad):
			x = Variable(x, requires_grad=True)
		logits = forward(model, x)
		max_logit, y = logits.max(1)
		x_grad, = torch.autograd.grad(max_logit.sum(), x, create_graph=self.requires_grad)
		return x_grad
class ScaledExplainerNoAbs(Explainer):
	def __init__(self, requires_grad=False):
		self.requires_grad = requires_grad
	def adjmax(self,x):
		return torch.where(x.max(1)[0]==0, torch.tensor(.1).cuda(), x.max(1)[0])
	def explain(self, model, x):
		if (not x.requires_grad):
			x = Variable(x, requires_grad=True)
		logits = forward(model, x)
		max_logit, y = logits.max(1)
		x_grad, = torch.autograd.grad(max_logit.sum(), x, create_graph=self.requires_grad)
		flat= x_grad.reshape(x_grad.shape[0],-1)
		flat = (flat.t()-flat.min(1)[0]).t()
		flattened_scaled = (flat.t()/self.adjmax(flat)).t()
		return flattened_scaled.reshape(x_grad.shape)
class BaseExplainerSquared(Explainer):
	def __init__(self, requires_grad=False):
		self.requires_grad = requires_grad
	def explain(self, model, x):
		if (not x.requires_grad):
			x = Variable(x, requires_grad=True)
		logits = forward(model, x)
		max_logit, y = logits.max(1)
		x_grad, = torch.autograd.grad(max_logit.sum(), x, create_graph=self.requires_grad)
		return x_grad * x_grad
class ScaledExplainerSquared(Explainer):
	def __init__(self, requires_grad=False):
		self.requires_grad = requires_grad
	def adjmax(self,x):
		return torch.where(x.max(1)[0]==0, torch.tensor(.1).cuda(), x.max(1)[0])
	def explain(self, model, x):
		if (not x.requires_grad):
			x = Variable(x, requires_grad=True)
		logits = forward(model, x)
		max_logit, y = logits.max(1)
		x_grad, = torch.autograd.grad(max_logit.sum(), x, create_graph=self.requires_grad)
		flat= x_grad.reshape(x_grad.shape[0],-1)
		squared = flat * flat
		flattened_scaled = (squared.t()/self.adjmax(squared)).t()
		return flattened_scaled.reshape(x_grad.shape)
def get_scale(flattened, pos):
	sz = flattened.nelement()
	if (sz == pos):
		return sz
	else:
		scale = flattened.topk(pos)[0][-1]
		bottomnorm = flattened.topk(sz-pos,largest=False)[0].sum()
		return pos + bottomnorm/scale

def contains_range(flattened, bottom, top, desired):
	bval = get_scale(flattened, bottom)
	tval = get_scale(flattened, top)
	if (bval <= desired and tval >= desired):
		if (bottom == top or bottom == top -1):
			return (bottom,top)
		else:
			return contains_range(flattened, bottom, int((top+bottom)/2), desired) or contains_range(flattened, int((top+bottom)/2), top, desired)
	else:
		return False

def scale_to_l1(flattened, desired):
	sz = flattened.nelement()
	rnge = contains_range(flattened, 1, flattened.nelement(), desired)
	if (rnge):
		contrib = desired-rnge[0]
		scale = contrib/flattened.topk(sz-rnge[0],largest=False)[0].sum()
		return (flattened*scale).clamp(max=1.0)
	else:
		contrib = desired
		scale = contrib/flattened.sum()
		return (flattened*scale)

class L0L1Explainer(Explainer):

	def __init__(self, l_0, l_1):
		self.l_0 = l_0
		self.l_1 = l_1
	def explain(self, model, x):
		x = Variable(x, requires_grad=True)
		logits = forward(model, x)
		max_logit, y = logits.max(1)
		x_grad, = torch.autograd.grad(max_logit.sum(), x)
		x_grad = torch.abs(x_grad)
		x_grad = x_grad.reshape(
			x_grad.shape[0],-1).scatter(
				1,
				torch.topk(x_grad.reshape(x_grad.shape[0],-1),x_grad[0].nelement()-self.l_0,largest=False,dim=1)[1],
				0).reshape(x_grad.shape)
		return torch.stack([scale_to_l1(i.reshape(-1), self.l_1).reshape(i.shape) for i in x_grad])

class ZeroOneExplainer(Explainer):

	def __init__(self, l_0):
		self.l_0 = l_0
	def explain(self, model, x):
		x = Variable(x, requires_grad=True)
		logits = forward(model, x)
		max_logit, y = logits.max(1)
		#print(max_logit)
		x_grad, = torch.autograd.grad(max_logit.sum(), x)
		x_grad = torch.abs(x_grad)
		x_sparsified = (x_grad.reshape(
			x_grad.shape[0],-1).scatter(
				1,
				torch.topk(x_grad.reshape(x_grad.shape[0],-1),x_grad[0].nelement()-self.l_0,largest=False,dim=1)[1],
				0).reshape(x_grad.shape) > 0).type(x_grad.type())
		return x_sparsified

class LinearExplainer(Explainer):

	def __init__(self, l_0):
		self.l_0 = l_0
	def explain(self, model, x):
		x = Variable(x, requires_grad=True)
		logits = forward(model, x)
		max_logit, y = logits.max(1)
		#print(max_logit)
		x_grad, = torch.autograd.grad(max_logit.sum(), x)
		x_grad = torch.abs(x_grad)
		x_ranks =  x_grad.reshape(x_grad.shape[0],-1).sort()[1].sort()[1].type(x_grad.type()) - (x_grad[0].nelement()-self.l_0)

		x_sparsified = (x_ranks + x_ranks.abs())/(2*(self.l_0-1))
		return x_sparsified.reshape(x_grad.shape)

class OrderExponentialExplainer(Explainer):

	def __init__(self, f):
		self.f = f
	def explain(self, model, x):
		x = Variable(x, requires_grad=True)
		logits = forward(model, x)
		max_logit, y = logits.max(1)
		x_grad, = torch.autograd.grad(max_logit, x)
		x_grad = torch.abs(x_grad)
		x_sparsified =  torch.exp(
			-x_grad.reshape(x_grad.shape[0],-1).sort(descending=True)[1].sort()[1].type(x_grad.type())/f
			).reshape(x_grad.shape)
		return x_sparsified



class SmoothGradBatchFastExplainer(Explainer):
	'''
	See https://arxiv.org/abs/1706.03825.
	'''
	def __init__(self, base_explainer=None, stdev=0.15, n_samples=8192, batch_size = 64):
		self.base_explainer = base_explainer
		self.stdev = stdev
		self.n_samples = n_samples
		self.batch_size = batch_size
	def explain(self, model, x):
		num_batches = self.n_samples/self.batch_size
		gradients = []
		for xi in x:
			total_gradients = 0
			for i in range(int(num_batches)):
				noise = torch.randn(tuple([self.batch_size]) + xi.shape ).cuda() * self.stdev
				x_var = noise + xi.clone()
				grad = self.base_explainer.explain(model, x_var)
				total_gradients += grad.sum(dim=0)
			total_gradients /= self.n_samples
			gradients.append(total_gradients)
		return torch.stack(gradients)

# Ghorbani's  L_inf attack projected onto L_2 ball
class PureGhorbaniAttacker(Attacker):
	def __init__(self, l2, explainer, iterations, stepsize, k):
		self.explainer = explainer
		self.iterations = iterations
		self.l2 = l2
		self.stepsize = stepsize
		self.k = k
	def obtain_gradient(self, model, x, target_mask):
		x = Variable(x, requires_grad = True)
		raw_saliency = self.explainer.explain(model, x)
		raw_saliency = raw_saliency.abs()
		flattened_raw_saliency = raw_saliency.reshape(raw_saliency.shape[0], -1)
		flattened_normalized_saliency = (flattened_raw_saliency.t()/flattened_raw_saliency.sum(dim=1)).t()
		target_function_itemized = (flattened_normalized_saliency*target_mask).sum(dim=1)
		target_function = target_function_itemized.sum()
		
		ret, = torch.autograd.grad(target_function,x, allow_unused=True)

		return ret, target_function_itemized

	def explain(self, model, x, classifier_model=None):
		if (classifier_model == None):
			classifier_model = model
		y_org = classifier_model(Variable(x)).max(1)[1].data
		orig_grad = torch.abs(self.explainer.explain(model, x))
		sz = x[0].nelement()
		top_k_mask = get_top_k_mask(orig_grad,self.k).reshape(x.shape[0],-1)
		x_curr = x.clone()
		best_scores = math.inf * torch.ones(x.shape[0]).type(x.type()).cuda()
		best_x = x.clone()
		fail_mask =  torch.ones(x.shape[0]).type('torch.ByteTensor').cuda()
		for i in range(self.iterations):
			current_x_valid = classifier_model(Variable(x_curr)).max(1)[1].data == y_org

			grad, val = self.obtain_gradient(model, x_curr, top_k_mask)
			best_mask = (current_x_valid * (val < best_scores)).type('torch.ByteTensor').cuda()
			fail_mask = fail_mask * (1-best_mask)
			best_scores[best_mask.nonzero()] = val[best_mask.nonzero()]
			#print(best_scores.sum())
			best_x[best_mask.nonzero()] = x_curr[best_mask.nonzero()]

			x_curr = x_curr.clone() - self.stepsize*torch.sign(grad)
			normalize_if_necessary(x_curr, x, self.l2)
		return best_x, fail_mask

# L_2 extension of Ghorbani attack

class EnhancedGhorbaniBatchL2Attacker(Attacker):
	def __init__(self, l2, explainer, search_iterations, optimize_iterations, restarts, k):
		self.explainer = explainer
		self.search_iterations = search_iterations
		self.optimize_iterations = optimize_iterations
		self.restarts = restarts

		self.l2 = l2
		self.k = k
	def obtain_gradient(self, model, x, target_mask):
		x = Variable(x, requires_grad = True)
		raw_saliency = self.explainer.explain(model, x)
		raw_saliency = raw_saliency.abs()
		flattened_raw_saliency = raw_saliency.reshape(raw_saliency.shape[0], -1)
		flattened_normalized_saliency = (flattened_raw_saliency.t()/flattened_raw_saliency.sum(dim=1)).t()
		target_function_itemized = (flattened_normalized_saliency*target_mask).sum(dim=1)
		target_function = target_function_itemized.sum()
		
		ret, = torch.autograd.grad(target_function,x, allow_unused=True)

		return ret, target_function_itemized

	def explain(self, model, x, classifier_model=None):
		if (classifier_model == None):
			classifier_model = model
		y_org = classifier_model(Variable(x)).max(1)[1].data
		orig_grad = torch.abs(self.explainer.explain(model, x))
		sz = x[0].nelement()
		top_k_mask = get_top_k_mask(orig_grad,self.k).reshape(x.shape[0],-1)
		x_curr = x.clone()
		best_score = math.inf * torch.ones(x.shape[0]).type(x.type()).cuda()
		best_x = x.clone()
		it = self.search_iterations * torch.ones(x.shape[0])
		for j in range(self.restarts):
			not_yet_worked = torch.ones(x.shape[0])
			while (sum(not_yet_worked*it) > 0):
				#print(sum(not_yet_worked))
				cycle = not_yet_worked * it
				it[cycle.nonzero()] = it[cycle.nonzero()] -1
				noise = torch.randn(x[cycle.nonzero()].squeeze(dim=1).shape).cuda()
				noise_flat =  noise.reshape(noise.shape[0],-1)
				noise = (noise_flat.t()/noise_flat.norm(dim=1)).t().reshape(noise.shape) * self.l2
				x_curr[cycle.nonzero()] = noise.unsqueeze(1)+ x[cycle.nonzero()].clone()
				normalize_if_necessary(x_curr, x, self.l2)
				y_model= classifier_model(Variable(x_curr[cycle.nonzero()].squeeze(dim=1))).max(1)[1].data
				not_yet_worked[cycle.nonzero()] = (y_org[cycle.nonzero()].squeeze(dim=1) != y_model).unsqueeze(1).type(not_yet_worked.type())
			if (sum(not_yet_worked) == x.shape[0]):
				continue
			step = None
			worked = (1-not_yet_worked).nonzero()
			for i in range(self.optimize_iterations):

				grad, val = self.obtain_gradient(model, x_curr[worked].squeeze(dim=1), top_k_mask[worked].squeeze(dim=1))

				best_x.data[worked[:,0][(val.unsqueeze(1) < best_score[worked]).squeeze(dim=1).nonzero()]]= x_curr[worked].squeeze(dim=1)[(val.unsqueeze(1) < best_score[worked]).squeeze(dim=1).nonzero()]

				best_score.data[worked[:,0][(val.unsqueeze(1) < best_score[worked]).squeeze(dim=1).nonzero()]] = val[(val.unsqueeze(1) < best_score[worked]).squeeze(dim=1).nonzero()]
				
				if (step is None):
					step = self.l2/grad.reshape(grad.shape[0],-1).norm(dim=1)

				x_new = x_curr[worked].squeeze(dim=1).clone() - (grad.reshape(grad.shape[0],-1).t()*step).t().reshape(grad.shape)
				normalize_if_necessary(x_new, x[worked].squeeze(dim=1), self.l2)
				die = 0
				refined_worked = classifier_model(Variable(x_new)).max(1)[1].data == y_org[worked].squeeze(dim=1)
				while (sum(1-refined_worked.int())):
					refined_not_worked = (~refined_worked).nonzero()
					if (die == 10):
						x_new[refined_not_worked] =  x_curr[worked].squeeze(dim=1)[refined_not_worked]
						step[refined_not_worked] = 0
						break
					step[refined_not_worked] = step[refined_not_worked]/2
					x_new[refined_not_worked] = x_curr[worked].squeeze(dim=1)[refined_not_worked].clone() - (grad[refined_not_worked].reshape(grad[refined_not_worked].shape[0],-1).t()*step[refined_not_worked].squeeze(dim=1)).t().reshape(grad[refined_not_worked].shape)
					normalize_if_necessary(x_new, x[worked].squeeze(dim=1), self.l2)

					die = die + 1
					refined_worked[refined_not_worked] = (classifier_model(Variable(x_new[refined_not_worked].squeeze(dim=1))).max(1)[1].data == y_org[worked].squeeze(dim=1)[refined_not_worked].squeeze(dim=1)).unsqueeze(1)
				step = step*(i+1)/(i+2)
				x_curr[worked[:,0][refined_worked.nonzero()]] = x_new[refined_worked.nonzero()]
		return best_x, (best_score == math.inf)



# L_2 extension of Ghorbani attack

class EnhancedGhorbaniSingleSampleL2Attacker(Attacker):
	def __init__(self, l2, explainer, search_iterations, optimize_iterations, restarts, k):
		self.explainer = explainer
		self.search_iterations = search_iterations
		self.optimize_iterations = optimize_iterations
		self.restarts = restarts

		self.l2 = l2
		self.k = k
	def obtain_gradient(self, model, x, target_mask):
		x = Variable(x, requires_grad = True)
		raw_saliency = self.explainer.explain(model, x)
		raw_saliency = raw_saliency.abs()
		flattened_raw_saliency = raw_saliency.reshape(raw_saliency.shape[0], -1)
		flattened_normalized_saliency = (flattened_raw_saliency.t()/flattened_raw_saliency.sum(dim=1)).t()
		target_function_itemized = (flattened_normalized_saliency*target_mask).sum(dim=1)
		target_function = target_function_itemized.sum()
		
		ret, = torch.autograd.grad(target_function,x, allow_unused=True)

		return ret, target_function_itemized

	def explain(self, model, x, classifier_model=None):
		if (classifier_model == None):
			classifier_model = model
		y_org = classifier_model(Variable(x)).max(1)[1].data
		orig_grad = torch.abs(self.explainer.explain(model, x))
		sz = x[0].nelement()
		top_k_mask = get_top_k_mask(orig_grad,self.k).reshape(x.shape[0],-1)
		x_curr = x.clone()
		best_score = math.inf
		best_x = x.clone()
		it = self.search_iterations
		for j in range(self.restarts):
			worked = 0
			while (it > 0):
				it = it -1
				noise = torch.randn(x.shape).cuda()
				noise_flat =  noise.reshape(noise.shape[0],-1)
				noise = (noise_flat.t()/noise_flat.norm(dim=1)).t().reshape(noise.shape) * self.l2
				x_curr = noise+ x
				y_model = classifier_model(Variable(x_curr)).max(1)[1].data
				if (y_org == y_model):
					worked = 1
					break
			if (not worked):
				continue
			step = None
			for i in range(self.optimize_iterations):


				grad, val = self.obtain_gradient(model, x_curr, top_k_mask)
				if (val[0] < best_score):
					best_score = val[0]
					best_x= x_curr
				print(best_score)
				if (step is None):
					step = self.l2/grad.norm()
				x_new = x_curr.clone() - step*grad
				normalize_if_necessary(x_new, x, self.l2)
				print(step)
				die = 0
				while (classifier_model(Variable(x_new)).max(1)[1].data != y_org):
					if (die == 200):
						break
					step = step/2
					x_new = x_curr.clone() - step*grad
					normalize_if_necessary(x_new, x, self.l2)
					die = die + 1
				if (die == 200):
					continue

				x_curr = x_new
		if (best_score == math.inf):
			return None
		return best_x


class SingleJumpL2FastAttackerWithFailure(Attacker):
	def __init__(self, l2, max_iterations):
		self.l2 = l2
		self.max_iterations = max_iterations
	def explain(self, model, x):
		y_org = model(Variable(x)).max(1)[1].data
		mask = torch.ones(x.shape[0]).cuda()
		accum = x.clone()
		for i in range(self.max_iterations):
			noise = torch.randn(x.shape).cuda()
			noise_flat =  noise.reshape(noise.shape[0],-1)
			noise = (noise_flat.t()/noise_flat.norm(dim=1)).t().reshape(noise.shape) * self.l2
			accum[mask.nonzero()] = noise[mask.nonzero()] + x[mask.nonzero()]
			clamp_batch(accum)
			y_model = model(Variable(accum)).max(1)[1].data
			mask = (y_org != y_model)
			check = mask.sum()
			if (check == 0):
				#print(check)
				return accum, mask
		check = mask.sum()
		#print(check)
		return accum, mask


