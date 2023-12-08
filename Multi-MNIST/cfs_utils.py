# coding:utf-8
import time
import random
import sys
import torch
import tqdm
import gc
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from models import *
from copy import deepcopy
from min_norm_solvers import MinNormSolver
from torch.nn.modules.loss import CrossEntropyLoss
from scipy.optimize import minimize, Bounds, minimize_scalar
import torch.nn.functional as F

# CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 50
num_init = 1
patch_size = 8
patches_nums = 16
tau = 0.1
img_size = [64, 64]
selected_dim1, selected_dim2 = img_size[0] // patch_size, img_size[1] // patch_size
k = 32


def break_point():
    sys.exit()


def control_seed(seed):
    torch.backends.cudnn.enabled = False
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def reshape_logits(discrete_logits, batch_size, dims, patch_size):
    discrete_logits = discrete_logits.type(torch.float32)
    discrete_logits = torch.unsqueeze(discrete_logits, dim=0)
    discrete_logits = discrete_logits.view(batch_size, dims[0], dims[1])
    discrete_logits = torch.unsqueeze(discrete_logits, dim=1)
    # discrete_logits = torch.abs(discrete_logits - 1)
    upsample_op = nn.Upsample(scale_factor=patch_size, mode='nearest')
    causal_logits = upsample_op(discrete_logits).to(device)

    return causal_logits





def generate_counterfactuals(X, selector, weights):
    batch_size = X.shape[0]
    softmax = nn.Softmax(dim=1)
    with torch.no_grad():
        logits = selector.forward(X)

    if torch.isnan(logits[0]).any():
        print("logits{} is nan in generate_counterfactuals".format(0))
    if torch.isnan(logits[1]).any():
        print("logits{} is nan in generate_counterfactuals".format(1))

    # causal_logits_for_task = [F.softmax(logits[i], dim=1) for i in range(2)]
    causal_logits_for_task = torch.stack((softmax(logits[0]), softmax(logits[1]),), dim=1)
    # print('causal_logits_for_task.size()', causal_logits_for_task.size())

    causal_logits, _ = torch.max(causal_logits_for_task, dim=1)
    # causal_logits = torch.zeros_like(causal_logits_for_task[0])
    non_causal_logits = torch.ones_like(causal_logits)
    # print('non_causal_logits.size()', non_causal_logits.size())
    non_causal_logits_task = [torch.ones_like(causal_logits) for i in range(2)]

    # for i in range(2):
    # causal_logits += causal_logits_for_task[i] * weights[i]

    # print('causal_logits.size()', causal_logits.size())
    # print('non_causal_logits.size()', non_causal_logits.size())

    non_causal_logits -= causal_logits

    for i in range(2):
        non_causal_logits_task[i] -= softmax(logits[i])

    X_Causal = torch.mul(X, causal_logits)
    X_non_Causal = torch.mul(X, non_causal_logits)
    X_Causal_for_tasks = [torch.mul(X, softmax(logits[i])) for i in range(2)]
    X_non_Causal_for_tasks = [torch.mul(X, softmax(logits[i])) for i in range(2)]

    del logits, causal_logits, non_causal_logits, causal_logits_for_task, non_causal_logits_task

    return X_Causal, X_non_Causal, X_Causal_for_tasks, X_non_Causal_for_tasks

   



'''
This function samples from a concrete distribution during training and while inference, it gives the indices of the top k logits
'''


def Sample_Concrete_for_MTL(Tau_coe, k, logits, train=True):
    d1, d2 = logits.shape[2], logits.shape[3]
    dims = d1 * d2
    batch_size = logits.shape[0]
    if train == True:
        softmax = nn.Softmax(dim=1)
        logits = logits.view(batch_size, -1).unsqueeze(1)  # .to('cpu')

        unif_shape = [batch_size, k, dims]
        uniform = (1 - 0) * torch.rand(unif_shape)
        # uniform = (1 - 0) * np.random.rand(unif_shape)
        gumbel = - torch.log(-torch.log(uniform)).to(device)
        noisy_logits = (gumbel + logits) / Tau_coe
        samples = softmax(noisy_logits)
        samples, _ = torch.max(samples, dim=1)

        samples = torch.reshape(samples, (batch_size, d1, d2))
        selected_subset = torch.unsqueeze(samples, dim=1)

        upsample_op = nn.Upsample(scale_factor=patch_size, mode='nearest')
        v = upsample_op(selected_subset)

        return v
    else:
        logits = torch.reshape(logits, [-1, dims])
        discrete_logits = torch.zeros(logits.shape[1])
        vals, ind = torch.topk(logits, k)
        discrete_logits[ind[0]] = 1
        discrete_logits = discrete_logits.type(torch.float32)  # change type to float32
        discrete_logits = torch.unsqueeze(discrete_logits, dim=0)
        return discrete_logits



def custom_loss(p_y_x, p_y_xs, batch_size):
    p_y_xs = p_y_xs.type(torch.float32)
    p_y_x = p_y_x.type(torch.float32)

    loss = 1 / torch.mean(torch.cosine_similarity(p_y_x, p_y_xs))
    # loss = torch.mean(torch.sum(p_y_x.view(batch_size, -1) * torch.log(p_y_xs.view(batch_size, -1)), dim=1))
    # loss_function = torch.nn.KLDivLoss(reduction='batchmean')

    return loss


def model_fit(x_pred, x_output):
    ce_loss = CrossEntropyLoss()
    loss = ce_loss(x_pred, x_output)

    return loss


# New mIoU and Acc. formula: accumulate every pixel and average across all pixels in all images
class ConfMatrix(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None

    def update(self, pred, target):
        n = self.num_classes
        if self.mat is None:
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=pred.device)
        with torch.no_grad():
            k = (target >= 0) & (target < n)
            inds = n * target[k].to(torch.int64) + pred[k]
            self.mat += torch.bincount(inds, minlength=n ** 2).reshape(n, n)

    def get_metrics(self):
        h = self.mat.float()
        acc = torch.diag(h).sum() / h.sum()
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        return torch.mean(iu).cpu(), acc.cpu()


def depth_error(x_pred, x_output):
    device = x_pred.device
    binary_mask = (torch.sum(x_output, dim=1) != 0).unsqueeze(1).to(device)
    x_pred_true = x_pred.masked_select(binary_mask)
    x_output_true = x_output.masked_select(binary_mask)
    abs_err = torch.abs(x_pred_true - x_output_true)
    rel_err = torch.abs(x_pred_true - x_output_true) / x_output_true
    return (torch.sum(abs_err) / torch.nonzero(binary_mask, as_tuple=False).size(0)).item(), \
           (torch.sum(rel_err) / torch.nonzero(binary_mask, as_tuple=False).size(0)).item()


def normal_error(x_pred, x_output):
    binary_mask = (torch.sum(x_output, dim=1) != 0)
    error = torch.acos(
        torch.clamp(torch.sum(x_pred * x_output, 1).masked_select(binary_mask), -1, 1)).detach().cpu().numpy()
    error = np.degrees(error)
    return np.mean(error), np.median(error), np.mean(error < 11.25), np.mean(error < 22.5), np.mean(error < 30)


def graddrop(grads):
    P = 0.5 * (1. + grads.sum(1) / (grads.abs().sum(1) + 1e-8))
    U = torch.rand_like(grads[:, 0])
    M = P.gt(U).view(-1, 1) * grads.gt(0) + P.lt(U).view(-1, 1) * grads.lt(0)
    g = (grads * M.float()).mean(1)
    return g


def mgd(grads):
    grads_cpu = grads.t().cpu()
    sol, min_norm = MinNormSolver.find_min_norm_element([
        grads_cpu[t] for t in range(grads.shape[-1])])
    w = torch.FloatTensor(sol).to(grads.device)
    g = grads.mm(w.view(-1, 1)).view(-1)
    return g


def pcgrad(grads, rng):
    grad_vec = grads.t()
    num_tasks = 2

    shuffled_task_indices = np.zeros((num_tasks, num_tasks - 1), dtype=int)
    for i in range(num_tasks):
        task_indices = np.arange(num_tasks)
        task_indices[i] = task_indices[-1]
        shuffled_task_indices[i] = task_indices[:-1]
        rng.shuffle(shuffled_task_indices[i])
    shuffled_task_indices = shuffled_task_indices.T

    normalized_grad_vec = grad_vec / (
            grad_vec.norm(dim=1, keepdim=True) + 1e-8
    )  # num_tasks x dim
    modified_grad_vec = deepcopy(grad_vec)
    for task_indices in shuffled_task_indices:
        normalized_shuffled_grad = normalized_grad_vec[
            task_indices
        ]  # num_tasks x dim
        dot = (modified_grad_vec * normalized_shuffled_grad).sum(
            dim=1, keepdim=True
        )  # num_tasks x dim
        modified_grad_vec -= torch.clamp_max(dot, 0) * normalized_shuffled_grad
    g = modified_grad_vec.mean(dim=0)
    return g


def cagrad(grads, alpha=0.5, rescale=0):
    g1 = grads[:, 0]
    g2 = grads[:, 1]

    g11 = g1.dot(g1).item()
    g12 = g1.dot(g2).item()
    g22 = g2.dot(g2).item()

    g0_norm = 0.5 * np.sqrt(g11 + g22 + 2 * g12)

    # want to minimize g_w^Tg_0 + c*||g_0||*||g_w||
    coef = alpha * g0_norm

    def obj(x):
        # g_w^T g_0: x*0.5*(g11+g22-2g12)+(0.5+x)*(g12-g22)+g22
        # g_w^T g_w: x^2*(g11+g22-2g12)+2*x*(g12-g22)+g22
        return coef * np.sqrt(x ** 2 * (g11 + g22 - 2 * g12) + 2 * x * (g12 - g22) + g22 + 1e-8) + 0.5 * x * (
                g11 + g22 - 2 * g12) + (0.5 + x) * (g12 - g22) + g22

    res = minimize_scalar(obj, bounds=(0, 1), method='bounded')
    x = res.x

    gw_norm = np.sqrt(x ** 2 * g11 + (1 - x) ** 2 * g22 + 2 * x * (1 - x) * g12 + 1e-8)
    lmbda = coef / (gw_norm + 1e-8)
    g = (0.5 + lmbda * x) * g1 + (0.5 + lmbda * (1 - x)) * g2  # g0 + lmbda*gw
    if rescale == 0:
        return g
    elif rescale == 1:
        return g / (1 + alpha ** 2)
    else:
        return g / (1 + alpha)


def grad2vec(m, grads, grad_dims, task):
    # store the gradients
    grads[:, task].fill_(0.0)
    cnt = 0
    for mm in m.shared_modules():
        for p in mm.parameters():
            grad = p.grad
            if grad is not None:
                grad_cur = grad.data.detach().clone()
                beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
                en = sum(grad_dims[:cnt + 1])
                grads[beg:en, task].copy_(grad_cur.data.view(-1))
            cnt += 1


def overwrite_grad(m, newgrad, grad_dims):
    newgrad = newgrad * 2  # to match the sum loss
    cnt = 0
    for mm in m.shared_modules():
        for param in mm.parameters():
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            this_grad = newgrad[beg: en].contiguous().view(param.data.size())
            param.grad = this_grad.data.clone()
            cnt += 1
