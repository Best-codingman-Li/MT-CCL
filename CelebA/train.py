# coding:utf-8
import gc
import sys
import pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
import torch.utils.data.sampler as sampler
from sklearn import metrics
from sklearn.preprocessing import label_binarize
from cfs_utils import *
from losses import *
from model_segnet_selector_mt import *
from model_segnet_selector_mt_for_celeba import *


def model_fit(x_pred, x_output):
    ce_loss = CrossEntropyLoss()
    loss = ce_loss(x_pred, x_output)

    return loss

def pretrain_cfs(train_loader, test_loader, SegNet_MTAN, optimizer, scheduler, num_epochs, start_epoch, task_nums):
    T = opt.temp

    avg_train_cost = np.zeros([num_epochs, task_nums], dtype=np.float32)
    avg_test_cost = np.zeros([num_epochs, task_nums], dtype=np.float32)
    lambda_weight = np.ones([task_nums, num_epochs])

    avg_train_loss_term = [np.zeros([num_epochs, 3], dtype=np.float32) for _ in range(task_nums)]
    lambda_weight_loss_term = [np.ones([3, num_epochs]) for _ in range(task_nums)]

    w = [0] * task_nums
    w_loss_term = [[0] * 3 for _ in range(task_nums)]
    train_batch, test_batch = len(train_loader), len(test_loader)

    # training and testing loop
    for epoch in range(start_epoch, num_epochs):
        start_time = time.time()
        train_cost = np.zeros(task_nums, dtype=np.float32)
        test_cost = np.zeros(task_nums, dtype=np.float32)
        train_acc, train_f1, train_rescall, train_precision = ([0] * task_nums for _ in range(4))
        test_acc, test_f1, test_rescall, test_precision = ([0] * task_nums for _ in range(4))

        # compute weights
        if epoch == start_epoch or epoch == start_epoch + 1:
            lambda_weight[:, epoch] = 1.0
            lambda_weight_loss_term[:][:,epoch] = 1.0
        else:
            for i in range(task_nums):
                w[i] = avg_train_cost[epoch - 1, i] / avg_train_cost[epoch - 2, i]
                for j in range(3):
                    w_loss_term[i][j] = avg_train_loss_term[i][epoch - 1, j] / avg_train_loss_term[i][epoch - 2, j]
            for j in range(task_nums):
                lambda_weight[j, epoch] = task_nums * np.exp(w[j] / T) / (sum(np.exp(w[i] / T) for i in range(task_nums)))
                for l in range(3):
                    lambda_weight_loss_term[j][l, epoch] = 3 * np.exp(w_loss_term[j][l] / T) / (sum(np.exp(w_loss_term[j][i] / T) for i in range(3)))

        SegNet_MTAN.train()
        train_dataset = iter(train_loader)

        # training step
        for b in range(train_batch):
            train_data, train_label = next(train_dataset)
            train_data, train_label = train_data.to(device), train_label.to(device)
            batch_size = train_data.size(0)

            train_pred, train_cf_pred, task_cf_combine_pred = SegNet_MTAN(train_data, training=True)

            # zero the parameter gradients
            optimizer.zero_grad()

            # optimization function
            task_cf_combine_loss = {}

            train_cf_loss = [model_fit(train_cf_pred[i], train_label[:, i]) for i in range(task_nums)]
            train_loss = [model_fit(train_pred[i], train_label[:, i]) for i in range(task_nums)]


            for i in range(task_nums):
                for j in range(i + 1, task_nums):
                    task_cf_combine_loss[(i, j)] = [model_fit(task_cf_combine_pred[(i, j)][0], train_label[:, i]),
                                     model_fit(task_cf_combine_pred[(i, j)][1], train_label[:, j])]

            temp_loss, cls_1, cls_2 = ([0] * task_nums for _ in range(3))

            Granger_dic = {}
            for i in range(task_nums):
                for j in range(i + 1, task_nums):
                    Granger_dic[(i, j)] = torch.exp((task_cf_combine_loss[(i, j)][1] - train_cf_loss[j]) / task_cf_combine_loss[(i, j)][1])
                    Granger_dic[(j, i)] = torch.exp((task_cf_combine_loss[(i, j)][0] - train_cf_loss[i]) / \
                                          task_cf_combine_loss[(i, j)][0])
            
            for i in range(task_nums):
                cls_1[i] = train_loss[i] / (train_loss[i] + train_cf_loss[i])
                granger_cls_j_i = 0
                for j in range(task_nums):
                    if i == j:
                        continue
                    granger_cls_j_i += (1 / Granger_dic[(j, i)]) * train_loss[j]

                cls_2[i] = train_loss[i] / (train_loss[i] + granger_cls_j_i)
                temp_loss[i] = lambda_weight_loss_term[i][0, epoch] * train_loss[i] + lambda_weight_loss_term[i][1, epoch] * cls_1[i] + lambda_weight_loss_term[i][2, epoch] * cls_2[i]
                avg_train_loss_term[i][epoch, 0] = train_loss[i].item()
                avg_train_loss_term[i][epoch, 1] = cls_1[i].item()
                avg_train_loss_term[i][epoch, 1] = cls_2[i].item()

            loss = sum([lambda_weight[i, epoch] * temp_loss[i] for i in range(task_nums)])

            loss.backward()
            optimizer.step()
            
            output = [train_pred[i].max(1, keepdim=True)[1] for i in range(task_nums)]
            

            
            for i in range(task_nums):
                train_label_i = train_label[:, i].view_as(output[i]).cpu()
                train_acc[i] += metrics.accuracy_score(train_label_i, output[i].cpu()) / train_batch
                train_f1[i] += metrics.f1_score(train_label_i, output[i].cpu(), average='micro') / train_batch
                train_precision[i] += metrics.precision_score(train_label_i, output[i].cpu(),
                                                              average='micro') / train_batch
                train_rescall[i] += metrics.recall_score(train_label_i, output[i].cpu(), average='micro') / train_batch

            for i in range(task_nums):
                train_cost[i] = train_loss[i].item()

            avg_train_cost[epoch, :] += train_cost[:] / train_batch

        avg_train_loss = sum(avg_train_cost[epoch, :]) / task_nums
        avg_train_acc = sum(train_acc) / task_nums
        avg_train_f1 = sum(train_f1) / task_nums
        avg_train_rescall = sum(train_rescall) / task_nums
        avg_train_precision = sum(train_precision) / task_nums
        t1 = time.time()
        print(
            'Epoch: {:04d} | train loss : {:.4f} | train_acc: {:.4f} | train_f1: {:.4f} | train_rescall: {:.4f} | train_precision: {:.4f} | time :{:.4f} '
            .format(epoch, avg_train_loss, avg_train_acc, avg_train_f1, avg_train_rescall, avg_train_precision, t1 - start_time))
        print('train_loss:', avg_train_cost[epoch, :])

        # test step
        SegNet_MTAN.eval()

        test_dataset = iter(test_loader)

        for b in range(test_batch):
            test_data, test_label = next(test_dataset)
            test_data, test_label = test_data.to(device), test_label.to(device)

            with torch.no_grad():
                test_pred = SegNet_MTAN(test_data, training=False)
                test_loss = [model_fit(test_pred[i], test_label[:, i]) for i in range(task_nums)]

                test_output = [test_pred[i].max(1, keepdim=True)[1] for i in range(task_nums)]

                
                for i in range(task_nums):
                    test_label_i = test_label[:, i].view_as(test_output[i]).cpu()
                    test_acc[i] += metrics.accuracy_score(test_label_i, test_output[i].cpu()) / test_batch
                    test_f1[i] += metrics.f1_score(test_label_i, test_output[i].cpu(), average='micro') / test_batch
                    test_precision[i] += metrics.precision_score(test_label_i, test_output[i].cpu(),
                                                                 average='micro') / test_batch
                    test_rescall[i] += metrics.recall_score(test_label_i, test_output[i].cpu(),
                                                            average='micro') / test_batch


                for j in range(task_nums):
                    test_cost[j] = test_loss[j].item()
                avg_test_cost[epoch, :] += test_cost[:] / test_batch
        scheduler.step()

        end_time = time.time()

        avg_test_loss = sum(avg_test_cost[epoch, :]) / task_nums
        avg_test_acc = sum(test_acc) / task_nums
        avg_test_f1 = sum(test_f1) / task_nums
        avg_test_rescall = sum(test_rescall) / task_nums
        avg_test_precision = sum(test_precision) / task_nums

        print(
            'Epoch: {:04d} | test loss : {:.4f} | test_acc: {:.4f} | test_f1: {:.4f} | test_rescall: {:.4f} | test_precision: {:.4f} | time :{:.4f} '
            .format(epoch, avg_test_loss, avg_test_acc, avg_test_f1, avg_test_rescall, avg_test_precision,
                    end_time - start_time))
        print('test_loss:', avg_test_cost[epoch, :])
        state = {'net': SegNet_MTAN.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
        torch.save(state, f"models/MT-CCL_K_Gumbel_Softmax_celeba.pt")



def main():
    parser = argparse.ArgumentParser(description='Multi-task: Attention Network')
    parser.add_argument('--method', default='mtan', type=str, help='which optimization algorithm to use')
    parser.add_argument('--weight', default='equal', type=str, help='multi-task weighting: equal, uncert, dwa')
    parser.add_argument('--dataroot', default='CelebA', type=str, help='dataset root')
    parser.add_argument('--temp', default=2.0, type=float, help='temperature for DWA (must be positive)')
    parser.add_argument('--alpha', default=0.5, type=float, help='the alpha')
    parser.add_argument('--lr', default=1e-4, type=float, help='the learning rate')
    parser.add_argument('--seed', default=0, type=int, help='control seed')
    parser.add_argument('--apply_augmentation', action='store_true', help='toggle to apply data augmentation on CelebA')
    opt = parser.parse_args()
    opt = parser.parse_args()

    # control seed
    torch.backends.cudnn.enabled = False
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    random.seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

    # load dataset
    img_root = '/code/CFS_DoubleMnist/img_align_celeba'
    image_text = '/code/CFS_DoubleMnist/Anno/list_attr_celeba.txt'
    train_text = '/code/CFS_DoubleMnist/Anno/train_attr_celeba.txt'
    test_text = '/code/CFS_DoubleMnist/Anno/test_attr_celeba.txt'
    batch_size = 64

    transform = transforms.Compose([
        transforms.Resize(40),
        transforms.CenterCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])

    train_dataset = myDataset(img_dir=img_root, img_txt=train_text, transform=transform)
    train_loader = Data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                   generator=torch.Generator(device='cuda'))
    test_dataset = myDataset(img_dir=img_root, img_txt=test_text, transform=transform)
    test_loader = Data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True,
                                  generator=torch.Generator(device='cuda'))


    SegNet_MTAN = SegNet_K_Gumbel_Softmax_TN(40, k, patch_size, tau, device).to(device)
    optimizer = optim.Adam(SegNet_MTAN.parameters(), lr=opt.lr)
    start_epoch = 0
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    num_epochs = 100
    pretrain_cfs(train_loader, test_loader, SegNet_MTAN, optimizer, scheduler, num_epochs, start_epoch, 40)


if __name__ == '__main__':
    main()
