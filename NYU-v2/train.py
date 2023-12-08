# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
import torch.utils.data.sampler as sampler
from cfs_utils import *
from cfs_utils import cagrad, grad2vec, overwrite_grad
from model_segnet_selector_mt import *
import torch.autograd as autograd


def graddrop(grads):
    P = 0.5 * (1. + grads.sum(1) / (grads.abs().sum(1)+1e-8))
    U = torch.rand_like(grads[:,0])
    M = P.gt(U).view(-1,1)*grads.gt(0) + P.lt(U).view(-1,1)*grads.lt(0)
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
    GG = grads.t().mm(grads).cpu() # [num_tasks, num_tasks]
    g0_norm = (GG.mean()+1e-8).sqrt() # norm of the average gradient

    x_start = np.ones(3) / 3
    bnds = tuple((0,1) for x in x_start)
    cons=({'type':'eq','fun':lambda x:1-sum(x)})
    A = GG.numpy()
    b = x_start.copy()
    c = (alpha*g0_norm+1e-8).item()
    def objfn(x):
        return (x.reshape(1,3).dot(A).dot(b.reshape(3, 1)) + c * np.sqrt(x.reshape(1,3).dot(A).dot(x.reshape(3,1))+1e-8)).sum()
    res = minimize(objfn, x_start, bounds=bnds, constraints=cons)
    w_cpu = res.x
    ww = torch.Tensor(w_cpu).to(grads.device)
    gw = (grads * ww.view(1, -1)).sum(1)
    gw_norm = gw.norm()
    lmbda = c / (gw_norm+1e-8)
    g = grads.mean(1) + lmbda * gw
    if rescale== 0:
        return g
    elif rescale == 1:
        return g / (1+alpha**2)
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
    newgrad = newgrad * 3 # to match the sum loss
    cnt = 0
    for mm in m.shared_modules():
        for param in mm.parameters():
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            this_grad = newgrad[beg: en].contiguous().view(param.data.size())
            param.grad = this_grad.data.clone()
            cnt += 1


# pre_training cfs_models
def cfs_cf_contrast(nyuv2_train_loader, nyuv2_test_loader, SegNet_MTAN, optimizer, scheduler, num_epochs, start_epoch):
    T = opt.temp
    alpha = opt.alpha
    method = opt.method

    rng = np.random.default_rng()
    grad_dims = []
    for mm in SegNet_MTAN.shared_modules():
        for param in mm.parameters():
            grad_dims.append(param.data.numel())
    grads = torch.Tensor(sum(grad_dims), 3).cuda()

    # variable for keeping track of best ph_acc across different iterations
    avg_cost = np.zeros([num_epochs, 24], dtype=np.float32)
    avg_loss_term = [np.zeros([num_epochs, 3], dtype=np.float32) for _ in range(3)]
    lambda_weight = np.ones([3, num_epochs])
    w_loss_term = [[0] * 3 for _ in range(3)]
    lambda_loss_term_weight = [np.ones([3, num_epochs]) for _ in range(3)]
    train_batch, test_batch = len(nyuv2_train_loader), len(nyuv2_test_loader)

    # training and testing loop
    for epoch in range(start_epoch, num_epochs):
        start_time = time.time()
        conf_mat = ConfMatrix(SegNet_MTAN.class_nb)
        cost = np.zeros(24, dtype=np.float32)

        # compute weights
        if epoch == start_epoch or epoch == start_epoch + 1:
            lambda_weight[:, epoch] = 1.0
            lambda_loss_term_weight[:][:,epoch] = 1.0
        else:
            w_1 = avg_cost[epoch - 1, 0] / avg_cost[epoch - 2, 0]
            w_2 = avg_cost[epoch - 1, 3] / avg_cost[epoch - 2, 3]
            w_3 = avg_cost[epoch - 1, 6] / avg_cost[epoch - 2, 6]
            for i in range(3):
                for j in range(3):
                    w_loss_term[i][j] = avg_loss_term[i][epoch - 1, j] / avg_loss_term[i][epoch - 2, j]
            lambda_weight[0, epoch] = 3 * np.exp(w_1 / T) / (np.exp(w_1 / T) + np.exp(w_2 / T) + np.exp(w_3 / T))
            lambda_weight[1, epoch] = 3 * np.exp(w_2 / T) / (np.exp(w_1 / T) + np.exp(w_2 / T) + np.exp(w_3 / T))
            lambda_weight[2, epoch] = 3 * np.exp(w_3 / T) / (np.exp(w_1 / T) + np.exp(w_2 / T) + np.exp(w_3 / T))

            for i in range(3):
                for j in range(3):
                    lambda_loss_term_weight[i][j,epoch] = 3 * np.exp(w_loss_term[i][j] / T) / sum(np.exp(w_loss_term[i][l] / T) for l in range(3))

        SegNet_MTAN.train()
        train_dataset = iter(nyuv2_train_loader)
        # training step
        for b in range(train_batch):
            train_data, train_label, train_depth, train_normal = train_dataset.next()
            train_data, train_label = train_data.to(device), train_label.long().to(device)
            train_depth, train_normal = train_depth.to(device), train_normal.to(device)
            batch_size = train_data.size(0)

            train_pred, train_cf_pred, t_1_2_cf_pred, t_1_3_cf_pred, t_2_3_cf_pred = SegNet_MTAN(train_data, training=True)
            # zero the parameter gradients
            optimizer.zero_grad()

            # optimization function
            train_loss = [model_fit(train_pred[0], train_label, 'semantic'),
                          model_fit(train_pred[1], train_depth, 'depth'),
                          model_fit(train_pred[2], train_normal, 'normal')]

            train_cf_loss = [model_fit(train_cf_pred[0], train_label, 'semantic'),
                             model_fit(train_cf_pred[1], train_depth, 'depth'),
                             model_fit(train_cf_pred[2], train_normal, 'normal')]

            t_union_cf_to_1_loss = [model_fit(t_1_2_cf_pred[0], train_label, 'semantic'), model_fit(t_1_3_cf_pred[0], train_label, 'semantic')]
            t_union_cf_to_2_loss = [model_fit(t_1_2_cf_pred[1], train_depth, 'depth'), model_fit(t_2_3_cf_pred[0], train_depth, 'depth')]
            t_union_cf_to_3_loss = [model_fit(t_1_3_cf_pred[1], train_normal, 'normal'), model_fit(t_2_3_cf_pred[1], train_normal, 'normal')]

            del train_cf_pred, t_1_2_cf_pred, t_1_3_cf_pred, t_2_3_cf_pred

            if torch.isnan(train_loss[0]).any():
                print("Epoch{}: batch_index:{} train_loss{} is nan".format(epoch, b, 0))
            if torch.isnan(train_loss[1]).any():
                print("Epoch{}: batch_index:{} train_loss{} is nan".format(epoch, b, 1))
            if torch.isnan(train_loss[2]).any():
                print("Epoch{}: batch_index:{} train_loss{} is nan".format(epoch, b, 2))

            if torch.isnan(train_cf_loss[0]).any():
                print("Epoch{}: batch_index:{} train_cf_loss{} is nan".format(epoch, b, 0))
            if torch.isnan(train_cf_loss[1]).any():
                print("Epoch{}: batch_index:{} train_cf_loss{} is nan".format(epoch, b, 1))
            if torch.isnan(train_cf_loss[2]).any():
                print("Epoch{}: batch_index:{} train_cf_loss{} is nan".format(epoch, b, 2))

            if torch.isnan(t_union_cf_to_1_loss[0]).any():
                print("Epoch{}: batch_index:{} t_union_cf_to_1_loss{} is nan".format(epoch, b, 0))
            if torch.isnan(t_union_cf_to_1_loss[1]).any():
                print("Epoch{}: batch_index:{} t_union_cf_to_1_loss{} is nan".format(epoch, b, 1))

            if torch.isnan(t_union_cf_to_2_loss[0]).any():
                print("Epoch{}: batch_index:{} t_union_cf_to_2_loss{} is nan".format(epoch, b, 0))
            if torch.isnan(t_union_cf_to_2_loss[1]).any():
                print("Epoch{}: batch_index:{} t_union_cf_to_2_loss{} is nan".format(epoch, b, 1))


            if torch.isnan(t_union_cf_to_3_loss[0]).any():
                print("Epoch{}: batch_index:{} t_union_cf_to_3_loss{} is nan".format(epoch, b, 0))
            if torch.isnan(t_union_cf_to_3_loss[1]).any():
                print("Epoch{}: batch_index:{} t_union_cf_to_3_loss{} is nan".format(epoch, b, 1))

            # compute Granger causal logits
            Granger_2_to_1 = torch.exp((t_union_cf_to_1_loss[0] - train_cf_loss[0]) / t_union_cf_to_1_loss[0])
            Granger_3_to_1 = torch.exp((t_union_cf_to_1_loss[1] - train_cf_loss[0]) / t_union_cf_to_1_loss[1])

            Granger_1_to_2 = torch.exp((t_union_cf_to_2_loss[0] - train_cf_loss[1]) / t_union_cf_to_2_loss[0])
            Granger_3_to_2 = torch.exp((t_union_cf_to_2_loss[1] - train_cf_loss[1]) / t_union_cf_to_2_loss[1])

            Granger_1_to_3 = torch.exp((t_union_cf_to_3_loss[0] - train_cf_loss[2]) / t_union_cf_to_3_loss[0])
            Granger_2_to_3 = torch.exp((t_union_cf_to_3_loss[1] - train_cf_loss[2]) / t_union_cf_to_3_loss[1])

            train_loss_tmp, contrast_loss_1, contrast_loss_2 = [0, 0, 0], [0, 0, 0], [0, 0, 0]

            for i in range(3):
                contrast_loss_1[i] = train_loss[i] / (train_loss[i] + train_cf_loss[i])
                #compute inter_task cls
                if i == 0:
                    granger_cls = (1 / Granger_2_to_1) * train_loss[1] + (1 / Granger_3_to_1) * train_loss[2]
                elif i == 1:
                    granger_cls = (1 / Granger_1_to_2) * train_loss[0] + (1 / Granger_3_to_2) * train_loss[2]
                elif i == 2:
                    granger_cls = (1 / Granger_1_to_3) * train_loss[0] + (1 / Granger_2_to_3) * train_loss[1]

                contrast_loss_2[i] =  train_loss[i] / (train_loss[i] + granger_cls)
                train_loss_tmp[i] = lambda_loss_term_weight[i][0,epoch] * train_loss[i] + lambda_loss_term_weight[i][1,epoch] * contrast_loss_1[i] + lambda_loss_term_weight[i][2,epoch] * contrast_loss_2[i]
                avg_loss_term[i][epoch, 0] = train_loss[i].item()
                avg_loss_term[i][epoch, 1] = contrast_loss_1[i].item()
                avg_loss_term[i][epoch, 2] = contrast_loss_2[i].item()

            if method == "graddrop":
                for i in range(3):
                    if i < 3:
                        train_loss_tmp[i].backward(retain_graph=True)
                    else:
                        train_loss_tmp[i].backward()
                    grad2vec(SegNet_MTAN, grads, grad_dims, i)
                    SegNet_MTAN.zero_grad_shared_modules()
                g = graddrop(grads)
                overwrite_grad(SegNet_MTAN, g, grad_dims)
                optimizer.step()
            elif method == "mgd":
                for i in range(3):
                    if i < 3:
                        train_loss_tmp[i].backward(retain_graph=True)
                    else:
                        train_loss_tmp[i].backward()
                    grad2vec(SegNet_MTAN, grads, grad_dims, i)
                    SegNet_MTAN.zero_grad_shared_modules()
                g = mgd(grads)
                overwrite_grad(SegNet_MTAN, g, grad_dims)
                optimizer.step()
            elif method == "pcgrad":
                for i in range(3):
                    if i < 3:
                        train_loss_tmp[i].backward(retain_graph=True)
                    else:
                        train_loss_tmp[i].backward()
                    grad2vec(SegNet_MTAN, grads, grad_dims, i)
                    SegNet_MTAN.zero_grad_shared_modules()
                g = pcgrad(grads, rng)
                overwrite_grad(SegNet_MTAN, g, grad_dims)
                optimizer.step()
            elif method == "cagrad":
                for i in range(3):
                    if i < 3:
                        train_loss_tmp[i].backward(retain_graph=True)
                    else:
                        train_loss_tmp[i].backward()
                    grad2vec(SegNet_MTAN, grads, grad_dims, i)
                    SegNet_MTAN.zero_grad_shared_modules()
                g = cagrad(grads, alpha, rescale=1)
                overwrite_grad(SegNet_MTAN, g, grad_dims)
                optimizer.step()
                
            elif method == "mtan":

                loss = sum([lambda_weight[i, epoch] * train_loss_tmp[i] for i in range(3)])
                loss.backward()
                optimizer.step()

            # accumulate label prediction for every pixel in training images
            conf_mat.update(train_pred[0].argmax(1).flatten(), train_label.flatten())

            cost[0] = train_loss_tmp[0].item()
            cost[3] = train_loss_tmp[1].item()
            cost[4], cost[5] = depth_error(train_pred[1], train_depth)
            cost[6] = train_loss_tmp[2].item()
            cost[7], cost[8], cost[9], cost[10], cost[11] = normal_error(train_pred[2], train_normal)
            avg_cost[epoch, :12] += cost[:12] / train_batch

        # compute mIoU and acc
        avg_cost[epoch, 1:3] = conf_mat.get_metrics()

        # test step

        SegNet_MTAN.eval()
        conf_mat = ConfMatrix(SegNet_MTAN.class_nb)
        test_dataset = iter(nyuv2_test_loader)
        for b in range(test_batch):
            test_data, test_label, test_depth, test_normal = test_dataset.next()
            test_data, test_label = test_data.to(device), test_label.long().to(device)
            test_depth, test_normal = test_depth.to(device), test_normal.to(device)

            with torch.no_grad():
                test_pred = SegNet_MTAN(test_data, training=False)
                test_loss = [model_fit(test_pred[0], test_label, 'semantic'),
                             model_fit(test_pred[1], test_depth, 'depth'),
                             model_fit(test_pred[2], test_normal, 'normal')]


            conf_mat.update(test_pred[0].argmax(1).flatten(), test_label.flatten())

            cost[12] = test_loss[0].item()
            cost[15] = test_loss[1].item()
            cost[16], cost[17] = depth_error(test_pred[1], test_depth)
            cost[18] = test_loss[2].item()
            cost[19], cost[20], cost[21], cost[22], cost[23] = normal_error(test_pred[2], test_normal)
            avg_cost[epoch, 12:] += cost[12:] / test_batch

            # compute mIoU and acc
        avg_cost[epoch, 13:15] = conf_mat.get_metrics()

        scheduler.step()
        end_time = time.time()
        print(
            'Epoch: {:04d} | TRAIN: {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} ||'
            'TEST: {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} | {:.4f}'.format(
                epoch,
                avg_cost[epoch, 0], avg_cost[epoch, 1], avg_cost[epoch, 2], avg_cost[epoch, 3], avg_cost[epoch, 4],
                avg_cost[epoch, 5],
                avg_cost[epoch, 6], avg_cost[epoch, 7], avg_cost[epoch, 8], avg_cost[epoch, 9], avg_cost[epoch, 10],
                avg_cost[epoch, 11],
                avg_cost[epoch, 12], avg_cost[epoch, 13], avg_cost[epoch, 14], avg_cost[epoch, 15],
                avg_cost[epoch, 16], avg_cost[epoch, 17],
                avg_cost[epoch, 18], avg_cost[epoch, 19], avg_cost[epoch, 20], avg_cost[epoch, 21],
                avg_cost[epoch, 22], avg_cost[epoch, 23], end_time - start_time))

        state = {'net': SegNet_MTAN.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
        torch.save(state, f"models/MT-CCL_k_GS.pt")


def main():
    parser = argparse.ArgumentParser(description='Multi-task: Split')
    parser.add_argument('--type', default='standard', type=str, help='split type: standard, wide, deep')
    parser.add_argument('--weight', default='equal', type=str, help='multi-task weighting: equal, uncert, dwa')
    parser.add_argument('--dataroot', default='nyuv2', type=str, help='dataset root')
    parser.add_argument('--method', default='mgd', type=str, help='optimization method')
    parser.add_argument('--temp', default=2.0, type=float, help='temperature for DWA (must be positive)')
    parser.add_argument('--alpha', default=0.5, type=float, help='the alpha')
    parser.add_argument('--lr', default=1e-3, type=float, help='the learning rate')
    parser.add_argument('--seed', default=0, type=int, help='the seed')
    parser.add_argument('--apply_augmentation', action='store_true', help='toggle to apply data augmentation on NYUv2')
    opt = parser.parse_args()

    # control seed
    torch.backends.cudnn.enabled = False
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    random.seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(
        'LOSS FORMAT: SEMANTIC_LOSS MEAN_IOU PIX_ACC | DEPTH_LOSS ABS_ERR REL_ERR | NORMAL_LOSS MEAN MED <11.25 <22.5 <30')

    # define dataset
    dataset_path = opt.dataroot
    if opt.apply_augmentation:
        nyuv2_train_set = NYUv2(root=dataset_path, train=True, augmentation=True)
        print('Applying data augmentation on NYUv2.')
    else:
        nyuv2_train_set = NYUv2(root=dataset_path, train=True)
        print('Standard training strategy without data augmentation.')

    nyuv2_test_set = NYUv2(root=dataset_path, train=False)

    batch_size = 2
    nyuv2_train_loader = torch.utils.data.DataLoader(
        dataset=nyuv2_train_set,
        batch_size=batch_size,
        shuffle=True)

    nyuv2_test_loader = torch.utils.data.DataLoader(
        dataset=nyuv2_test_set,
        batch_size=batch_size,
        shuffle=False)


      # define model, optimiser and scheduler
    SegNet_MTAN = SegNet_Gumbel_Softmax(k, patch_size, tau, device).to(device)
    print("k=", k)
    print('method:', opt.method)
    optimizer = optim.Adam(SegNet_MTAN.parameters(), lr=opt.lr, eps=1e-4)
    start_epoch = 0
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    num_epochs = 200
    print("start_epoch", start_epoch)

    print('Parameter Space: ABS: {:.1f}, REL: {:.4f}'.format(count_parameters(SegNet_MTAN),
                                                             count_parameters(SegNet_MTAN) / 24981069))

    cfs_cf_contrast(nyuv2_train_loader, nyuv2_test_loader, SegNet_MTAN, optimizer, scheduler, num_epochs, start_epoch)


if __name__ == '__main__':
    main()
