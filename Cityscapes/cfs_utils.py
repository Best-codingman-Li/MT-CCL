# importing local libraries
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
from scipy.optimize import minimize, Bounds, minimize_scalar
import torch.nn.functional as F

# GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 200
patch_size = 16
patches_nums = 32
tau = 0.1
img_size = [288, 384]
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


def custom_loss(p_y_xs, p_y_x, task_type):
    p_y_xs = p_y_xs.type(torch.float32)
    p_y_x = p_y_x.type(torch.float32)

    loss_function = torch.nn.KLDivLoss(reduction='batchmean')

    if task_type == 'semantic':
        #loss = loss_function(p_y_xs, p_y_x)
        loss = loss_function(F.log_softmax(p_y_xs, dim=1), F.softmax(p_y_x, dim=1))
    if task_type == 'depth':
        loss = loss_function(F.log_softmax(p_y_xs, dim=1), F.softmax(p_y_x, dim=1))
    if task_type == 'normal':
        loss = loss_function(F.log_softmax(p_y_xs, dim=1), F.softmax(p_y_x, dim=1))

    return loss


def model_fit(x_pred, x_output, task_type):
    device = x_pred.device

    # binary mark to mask out undefined pixel space
    binary_mask = (torch.sum(x_output, dim=1) != 0).float().unsqueeze(1).to(device)

    if task_type == 'semantic':
        # semantic loss: depth-wise cross entropy
        loss = F.nll_loss(x_pred, x_output, ignore_index=-1)

    if task_type == 'depth':
        # depth loss: l1 norm
        loss = torch.sum(torch.abs(x_pred - x_output) * binary_mask) / torch.nonzero(binary_mask, as_tuple=False).size(
            0)

    if task_type == 'normal':
        # normal loss: dot product
        loss = 1 - torch.sum((x_pred * x_output) * binary_mask) / torch.nonzero(binary_mask, as_tuple=False).size(0)

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


def multi_task_trainer(train_loader, test_loader, multi_task_model, device, optimizer, scheduler, opt,
                       total_epoch=200):
    start_time = time.time()
    train_batch = len(train_loader)
    test_batch = len(test_loader)
    T = opt.temp
    avg_cost = np.zeros([total_epoch, 24], dtype=np.float32)
    lambda_weight = np.ones([3, total_epoch])
    for index in range(total_epoch):
        epoch_start_time = time.time()
        cost = np.zeros(24, dtype=np.float32)

        # apply Dynamic Weight Average
        if opt.weight == 'dwa':
            if index == 0 or index == 1:
                lambda_weight[:, index] = 1.0
            else:
                w_1 = avg_cost[index - 1, 0] / avg_cost[index - 2, 0]
                w_2 = avg_cost[index - 1, 3] / avg_cost[index - 2, 3]
                w_3 = avg_cost[index - 1, 6] / avg_cost[index - 2, 6]
                lambda_weight[0, index] = 3 * np.exp(w_1 / T) / (np.exp(w_1 / T) + np.exp(w_2 / T) + np.exp(w_3 / T))
                lambda_weight[1, index] = 3 * np.exp(w_2 / T) / (np.exp(w_1 / T) + np.exp(w_2 / T) + np.exp(w_3 / T))
                lambda_weight[2, index] = 3 * np.exp(w_3 / T) / (np.exp(w_1 / T) + np.exp(w_2 / T) + np.exp(w_3 / T))

        # iteration for all batches
        multi_task_model.train()
        train_dataset = iter(train_loader)
        conf_mat = ConfMatrix(multi_task_model.class_nb)
        for k in range(train_batch):
            train_data, train_label, train_depth, train_normal = train_dataset.next()
            train_data, train_label = train_data.to(device), train_label.long().to(device)
            train_depth, train_normal = train_depth.to(device), train_normal.to(device)

            train_pred, logsigma = multi_task_model(train_data)

            optimizer.zero_grad()
            train_loss = [model_fit(train_pred[0], train_label, 'semantic'),
                          model_fit(train_pred[1], train_depth, 'depth'),
                          model_fit(train_pred[2], train_normal, 'normal')]

            if opt.weight == 'equal' or opt.weight == 'dwa':
                loss = sum([lambda_weight[i, index] * train_loss[i] for i in range(3)])
                # loss = sum([w[i] * train_loss[i] for i in range(3)])
            else:
                loss = sum(1 / (2 * torch.exp(logsigma[i])) * train_loss[i] + logsigma[i] / 2 for i in range(3))

            loss.backward()
            optimizer.step()

            # accumulate label prediction for every pixel in training images
            conf_mat.update(train_pred[0].argmax(1).flatten(), train_label.flatten())

            cost[0] = train_loss[0].item()
            cost[3] = train_loss[1].item()
            cost[4], cost[5] = depth_error(train_pred[1], train_depth)
            cost[6] = train_loss[2].item()
            cost[7], cost[8], cost[9], cost[10], cost[11] = normal_error(train_pred[2], train_normal)
            avg_cost[index, :12] += cost[:12] / train_batch

        # compute mIoU and acc
        avg_cost[index, 1:3] = conf_mat.get_metrics()

        # evaluating test data
        multi_task_model.eval()
        conf_mat = ConfMatrix(multi_task_model.class_nb)
        with torch.no_grad():  # operations inside don't track history
            test_dataset = iter(test_loader)
            for k in range(test_batch):
                test_data, test_label, test_depth, test_normal = test_dataset.next()
                test_data, test_label = test_data.to(device), test_label.long().to(device)
                test_depth, test_normal = test_depth.to(device), test_normal.to(device)

                test_pred, _ = multi_task_model(test_data)
                test_loss = [model_fit(test_pred[0], test_label, 'semantic'),
                             model_fit(test_pred[1], test_depth, 'depth'),
                             model_fit(test_pred[2], test_normal, 'normal')]

                conf_mat.update(test_pred[0].argmax(1).flatten(), test_label.flatten())

                cost[12] = test_loss[0].item()
                cost[15] = test_loss[1].item()
                cost[16], cost[17] = depth_error(test_pred[1], test_depth)
                cost[18] = test_loss[2].item()
                cost[19], cost[20], cost[21], cost[22], cost[23] = normal_error(test_pred[2], test_normal)
                avg_cost[index, 12:] += cost[12:] / test_batch

            # compute mIoU and acc
            avg_cost[index, 13:15] = conf_mat.get_metrics()

        scheduler.step()
        epoch_end_time = time.time()
        print(
            'Epoch: {:04d} | TRAIN: {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} ||'
            'TEST: {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} | {:.4f}'
                .format(index, avg_cost[index, 0], avg_cost[index, 1], avg_cost[index, 2], avg_cost[index, 3],
                        avg_cost[index, 4], avg_cost[index, 5], avg_cost[index, 6], avg_cost[index, 7],
                        avg_cost[index, 8],
                        avg_cost[index, 9], avg_cost[index, 10], avg_cost[index, 11], avg_cost[index, 12],
                        avg_cost[index, 13],
                        avg_cost[index, 14], avg_cost[index, 15], avg_cost[index, 16], avg_cost[index, 17],
                        avg_cost[index, 18],
                        avg_cost[index, 19], avg_cost[index, 20], avg_cost[index, 21], avg_cost[index, 22],
                        avg_cost[index, 23],
                        epoch_end_time - epoch_start_time))
    end_time = time.time()
    print("Training time: ", end_time - start_time)


''' ===== multi task MGD trainer ==== '''


def multi_task_mgd_trainer(train_loader, test_loader, multi_task_model, device,
                           optimizer, scheduler, opt,
                           total_epoch=200, method='sumloss', alpha=0.5, seed=0):
    start_time = time.time()

    def graddrop(grads):
        P = 0.5 * (1. + grads.sum(1) / (grads.abs().sum(1) + 1e-8))
        U = torch.rand_like(grads[:, 0])
        M = P.gt(U).view(-1, 1) * grads.gt(0) + P.lt(U).view(-1, 1) * grads.lt(0)
        g = (grads * M.float()).mean(1)
        return g

    def mgd(grads):
        grads_cpu = grads.t().cpu()
        sol, min_norm = MinNormSolver.find_min_norm_element([grads_cpu[t] for t in range(grads.shape[-1])])
        w = torch.FloatTensor(sol).to(grads.device)
        g = grads.mm(w.view(-1, 1)).view(-1)
        return g

    def pcgrad(grads, rng):
        grad_vec = grads.t()
        num_tasks = 3

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

    def cagrad(grads, alpha=0.5, rescale=1):
        GG = grads.t().mm(grads).cpu()  # [num_tasks, num_tasks]
        g0_norm = (GG.mean() + 1e-8).sqrt()  # norm of the average gradient

        x_start = np.ones(3) / 3
        bnds = tuple((0, 1) for x in x_start)
        cons = ({'type': 'eq', 'fun': lambda x: 1 - sum(x)})
        A = GG.numpy()
        b = x_start.copy()
        c = (alpha * g0_norm + 1e-8).item()

        def objfn(x):
            return (x.reshape(1, 3).dot(A).dot(b.reshape(3, 1)) + c * np.sqrt(
                x.reshape(1, 3).dot(A).dot(x.reshape(3, 1)) + 1e-8)).sum()

        res = minimize(objfn, x_start, bounds=bnds, constraints=cons)
        w_cpu = res.x
        ww = torch.Tensor(w_cpu).to(grads.device)
        gw = (grads * ww.view(1, -1)).sum(1)
        gw_norm = gw.norm()
        lmbda = c / (gw_norm + 1e-8)
        g = grads.mean(1) + lmbda * gw
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
        newgrad = newgrad * 3  # to match the sum loss
        cnt = 0
        for mm in m.shared_modules():
            for param in mm.parameters():
                beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
                en = sum(grad_dims[:cnt + 1])
                this_grad = newgrad[beg: en].contiguous().view(param.data.size())
                param.grad = this_grad.data.clone()
                cnt += 1

    rng = np.random.default_rng()
    grad_dims = []
    for mm in multi_task_model.shared_modules():
        for param in mm.parameters():
            grad_dims.append(param.data.numel())
    grads = torch.Tensor(sum(grad_dims), 3).cuda()

    # begin next training on train_data and test_data
    train_batch = len(train_loader)
    test_batch = len(test_loader)
    T = opt.temp
    avg_cost = np.zeros([total_epoch, 24], dtype=np.float32)
    lambda_weight = np.ones([3, total_epoch])

    neg_trace = []
    obj_trace = []
    for index in range(total_epoch):
        epoch_start_time = time.time()
        cost = np.zeros(24, dtype=np.float32)

        # apply Dynamic Weight Average
        if opt.weight == 'dwa':
            if index == 0 or index == 1:
                lambda_weight[:, index] = 1.0
            else:
                w_1 = avg_cost[index - 1, 0] / avg_cost[index - 2, 0]
                w_2 = avg_cost[index - 1, 3] / avg_cost[index - 2, 3]
                w_3 = avg_cost[index - 1, 6] / avg_cost[index - 2, 6]
                lambda_weight[0, index] = 3 * np.exp(w_1 / T) / (np.exp(w_1 / T) + np.exp(w_2 / T) + np.exp(w_3 / T))
                lambda_weight[1, index] = 3 * np.exp(w_2 / T) / (np.exp(w_1 / T) + np.exp(w_2 / T) + np.exp(w_3 / T))
                lambda_weight[2, index] = 3 * np.exp(w_3 / T) / (np.exp(w_1 / T) + np.exp(w_2 / T) + np.exp(w_3 / T))

        # iteration for all batches
        multi_task_model.train()
        train_dataset = iter(train_loader)
        conf_mat = ConfMatrix(multi_task_model.class_nb)
        for k in range(train_batch):
            train_data, train_label, train_depth, train_normal = train_dataset.next()
            train_data, train_label = train_data.to(device), train_label.long().to(device)
            train_depth, train_normal = train_depth.to(device), train_normal.to(device)

            train_pred, logsigma = multi_task_model(train_data)

            train_loss = [model_fit(train_pred[0], train_label, 'semantic'),
                          model_fit(train_pred[1], train_depth, 'depth'),
                          model_fit(train_pred[2], train_normal, 'normal')]

            train_loss_tmp = [0, 0, 0]

            if opt.weight == 'equal' or opt.weight == 'dwa':
                for i in range(3):
                    train_loss_tmp[i] = train_loss[i] * lambda_weight[i, index]
            else:
                for i in range(3):
                    train_loss_tmp[i] = 1 / (2 * torch.exp(logsigma[i])) * train_loss[i] + logsigma[i] / 2

            optimizer.zero_grad()
            if method == "graddrop":
                for i in range(3):
                    if i < 3:
                        train_loss_tmp[i].backward(retain_graph=True)
                    else:
                        train_loss_tmp[i].backward()
                    grad2vec(multi_task_model, grads, grad_dims, i)
                    multi_task_model.zero_grad_shared_modules()
                g = graddrop(grads)
                overwrite_grad(multi_task_model, g, grad_dims)
                optimizer.step()
            elif method == "mgd":
                for i in range(3):
                    if i < 3:
                        train_loss_tmp[i].backward(retain_graph=True)
                    else:
                        train_loss_tmp[i].backward()
                    grad2vec(multi_task_model, grads, grad_dims, i)
                    multi_task_model.zero_grad_shared_modules()
                g = mgd(grads)
                overwrite_grad(multi_task_model, g, grad_dims)
                optimizer.step()
            elif method == "pcgrad":
                for i in range(3):
                    if i < 3:
                        train_loss_tmp[i].backward(retain_graph=True)
                    else:
                        train_loss_tmp[i].backward()
                    grad2vec(multi_task_model, grads, grad_dims, i)
                    multi_task_model.zero_grad_shared_modules()
                g = pcgrad(grads, rng)
                overwrite_grad(multi_task_model, g, grad_dims)
                optimizer.step()
            elif method == "cagrad":
                for i in range(3):
                    if i < 3:
                        train_loss_tmp[i].backward(retain_graph=True)
                    else:
                        train_loss_tmp[i].backward()
                    grad2vec(multi_task_model, grads, grad_dims, i)
                    multi_task_model.zero_grad_shared_modules()
                g = cagrad(grads, alpha, rescale=1)
                overwrite_grad(multi_task_model, g, grad_dims)
                optimizer.step()

            # accumulate label prediction for every pixel in training images
            conf_mat.update(train_pred[0].argmax(1).flatten(), train_label.flatten())

            cost[0] = train_loss[0].item()
            cost[3] = train_loss[1].item()
            cost[4], cost[5] = depth_error(train_pred[1], train_depth)
            cost[6] = train_loss[2].item()
            cost[7], cost[8], cost[9], cost[10], cost[11] = normal_error(train_pred[2], train_normal)
            avg_cost[index, :12] += cost[:12] / train_batch

        # compute mIoU and acc
        avg_cost[index, 1:3] = conf_mat.get_metrics()

        # evaluating test data
        multi_task_model.eval()
        conf_mat = ConfMatrix(multi_task_model.class_nb)
        with torch.no_grad():  # operations inside don't track history
            test_dataset = iter(test_loader)
            for k in range(test_batch):
                test_data, test_label, test_depth, test_normal = test_dataset.next()
                test_data, test_label = test_data.to(device), test_label.long().to(device)
                test_depth, test_normal = test_depth.to(device), test_normal.to(device)

                test_pred, _ = multi_task_model(test_data)
                test_loss = [model_fit(test_pred[0], test_label, 'semantic'),
                             model_fit(test_pred[1], test_depth, 'depth'),
                             model_fit(test_pred[2], test_normal, 'normal')]

                conf_mat.update(test_pred[0].argmax(1).flatten(), test_label.flatten())

                cost[12] = test_loss[0].item()
                cost[15] = test_loss[1].item()
                cost[16], cost[17] = depth_error(test_pred[1], test_depth)
                cost[18] = test_loss[2].item()
                cost[19], cost[20], cost[21], cost[22], cost[23] = normal_error(test_pred[2], test_normal)
                avg_cost[index, 12:] += cost[12:] / test_batch

            # compute mIoU and acc
            avg_cost[index, 13:15] = conf_mat.get_metrics()

        scheduler.step()
        if method == "mean":
            torch.save(torch.Tensor(neg_trace), "trace.pt")

        if "debug" in method:
            torch.save(torch.Tensor(obj_trace), f"{method}_obj.pt")

        epoch_end_time = time.time()
        print(
            'Epoch: {:04d} | TRAIN: {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} ||'
            'TEST: {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} | {:.4f}'
                .format(index, avg_cost[index, 0], avg_cost[index, 1], avg_cost[index, 2], avg_cost[index, 3],
                        avg_cost[index, 4], avg_cost[index, 5], avg_cost[index, 6], avg_cost[index, 7],
                        avg_cost[index, 8],
                        avg_cost[index, 9], avg_cost[index, 10], avg_cost[index, 11], avg_cost[index, 12],
                        avg_cost[index, 13],
                        avg_cost[index, 14], avg_cost[index, 15], avg_cost[index, 16], avg_cost[index, 17],
                        avg_cost[index, 18],
                        avg_cost[index, 19], avg_cost[index, 20], avg_cost[index, 21], avg_cost[index, 22],
                        avg_cost[index, 23], epoch_end_time - epoch_start_time))
        if "cagrad" in method:
            torch.save(multi_task_model.state_dict(), f"models/{method}-{opt.weight}-{alpha}-{seed}.pt")
        else:
            torch.save(multi_task_model.state_dict(), f"models/{method}-{opt.weight}-{seed}.pt")
    end_time = time.time()
    print("Training time: ", end_time - start_time)


def test_mtl(test_loader, multi_task_model, device, scheduler, opt, total_epoch):
    method = opt.method
    alpha = opt.alpha
    # evaluating test data
    test_batch = len(test_loader)
    avg_cost = np.zeros([total_epoch, 6], dtype=np.float32)
    for index in range(total_epoch):
        t0 = time.time()
        multi_task_model.eval()
        conf_mat = ConfMatrix(multi_task_model.class_nb)
        cost = np.zeros(6, dtype=np.float32)
        with torch.no_grad():  # operations inside don't track history
            test_dataset = iter(test_loader)
            for k in range(test_batch):
                test_data, test_label, test_depth = test_dataset.next()
                test_data, test_label = test_data.to(device), test_label.long().to(device)
                test_depth = test_depth.to(device)

                test_pred, _ = multi_task_model(test_data)
                test_loss = [model_fit(test_pred[0], test_label, 'semantic'),
                             model_fit(test_pred[1], test_depth, 'depth')]

                conf_mat.update(test_pred[0].argmax(1).flatten(), test_label.flatten())

                cost[0] = test_loss[0].item()
                cost[3] = test_loss[1].item()
                cost[4], cost[5] = depth_error(test_pred[1], test_depth)
                avg_cost[index, 6:] += cost[:] / test_batch

            # compute mIoU and acc
            avg_cost[index, 7:9] = conf_mat.get_metrics()

        scheduler.step()
        t1 = time.time()
        print(
            'Epoch: {:04d} | TEST: {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} | TIME: {:.4f}'
                .format(index, avg_cost[index, 0], avg_cost[index, 1], avg_cost[index, 2], avg_cost[index, 3],
                        avg_cost[index, 4], avg_cost[index, 5], t1 - t0))
        torch.save(multi_task_model.state_dict(), f"models/cfs_test-{method}-{opt.weight}-{alpha}-{opt.seed}.pt")


def train_basemodel(trainloader, bb_model, LossFunc, optimizer, num_epochs, batch_size):
    # training loop
    for epoch in range(num_epochs):
        with tqdm(trainloader, unit="batch") as tepoch:
            for data, target in tepoch:
                data = data.to(device)
                target = target.to(device)
                tepoch.set_description("Epoch " + str(epoch))

                # data, target = data.to(device), target.to(device)
                optimizer.zero_grad()

                outputs = bb_model(data)
                loss = LossFunc(outputs, target)

                predictions = outputs.argmax(dim=1, keepdim=True).squeeze()
                correct = (predictions == target).sum().item()
                accuracy = correct / batch_size

                loss.backward()
                optimizer.step()

                tepoch.set_postfix(loss=loss.item(), accuracy=100. * accuracy)

    # uncomment to save the model
    torch.save(bb_model, 'mnist_model.pt')


def test_basemodel(valloader, bb_model):
    # testing the black box model performance on the entire validation dataset
    correct_count, all_count = 0, 0
    for images, labels in valloader:
        for i in range(len(labels)):
            img = images[i].to(device)
            img = img.unsqueeze(0)
            with torch.no_grad():
                out = bb_model(img)

            pred_label = torch.argmax(out)
            true_label = labels.numpy()[i]
            # true_label = label.numpy()
            if (true_label == pred_label):
                correct_count += 1
            all_count += 1

    print("Number Of Images Tested =", all_count)
    print("Model Accuracy =", (correct_count / all_count))


def cagrad(grads, alpha=0.5, rescale=1):
    GG = grads.t().mm(grads).cpu()  # [num_tasks, num_tasks]
    g0_norm = (GG.mean() + 1e-8).sqrt()  # norm of the average gradient

    x_start = np.ones(3) / 3
    bnds = tuple((0, 1) for x in x_start)
    cons = ({'type': 'eq', 'fun': lambda x: 1 - sum(x)})
    A = GG.numpy()
    b = x_start.copy()
    c = (alpha * g0_norm + 1e-8).item()

    def objfn(x):
        return (x.reshape(1, 3).dot(A).dot(b.reshape(3, 1)) + c * np.sqrt(
            x.reshape(1, 3).dot(A).dot(x.reshape(3, 1)) + 1e-8)).sum()

    res = minimize(objfn, x_start, bounds=bnds, constraints=cons)
    w_cpu = res.x
    ww = torch.Tensor(w_cpu).to(grads.device)
    gw = (grads * ww.view(1, -1)).sum(1)
    gw_norm = gw.norm()
    lmbda = c / (gw_norm + 1e-8)
    g = grads.mean(1) + lmbda * gw
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
    newgrad = newgrad * 3  # to match the sum loss
    cnt = 0
    for mm in m.shared_modules():
        for param in mm.parameters():
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            this_grad = newgrad[beg: en].contiguous().view(param.data.size())
            param.grad = this_grad.data.clone()
            cnt += 1
