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
from cfs_utils import *
from losses import *
from model_segnet_selector_mt import *


def cfs_cf_contrast(train_loader, test_loader, SegNet_MTAN, optimizer, scheduler, num_epochs, start_epoch):
    T = opt.temp
    alpha = opt.alpha

    avg_cost = np.zeros([num_epochs, 8], dtype=np.float32)
    avg_loss_term = [np.zeros([num_epochs, 3], dtype=np.float32) for _ in range(2)]
    lambda_weight = np.ones([2, num_epochs])
    lambda_lossterm_weight = [np.ones([3, num_epochs]) for i in range(2)]
    train_batch, test_batch = len(train_loader), len(test_loader)
    w_loss_term = [[0] * 3 for _ in range(2)]
    # training and testing loop
    for epoch in range(start_epoch, num_epochs):
        start_time = time.time()
        cost = np.zeros(8, dtype=np.float32)

        # compute weights
        if epoch == start_epoch or epoch == start_epoch + 1:
            lambda_weight[:, epoch] = 1.0
            lambda_lossterm_weight[:][:,epoch] = 1.0
        else:
            w_1 = avg_cost[epoch - 1, 0] / avg_cost[epoch - 2, 0]
            w_2 = avg_cost[epoch - 1, 2] / avg_cost[epoch - 2, 2]

            for i in range(2):
                for j in range(3):
                    w_loss_term[i][j] = avg_loss_term[i][epoch - 1, j] / avg_loss_term[i][epoch - 2, j]

            lambda_weight[0, epoch] = 2 * np.exp(w_1 / T) / (np.exp(w_1 / T) + np.exp(w_2 / T))
            lambda_weight[1, epoch] = 2 * np.exp(w_2 / T) / (np.exp(w_1 / T) + np.exp(w_2 / T))

            for i in range(2):
                for j in range(3):
                    lambda_lossterm_weight[i][j, epoch] = 3 * np.exp(w_loss_term[i][j] / T) / sum(np.exp(w_loss_term[i][l] / T) for l in range(3))


        SegNet_MTAN.train()
        train_dataset = iter(train_loader)
        correct1_train, correct2_train = 0, 0
        # training step
        for b in range(train_batch):
            train_data, train_label = train_dataset.next()
            train_data, train_label = train_data.to(device), train_label.to(device)
            batch_size = train_data.size(0)

            train_pred, train_cf_pred, t_1_2_cf_pred = SegNet_MTAN(train_data, training=True)

            # zero the parameter gradients
            optimizer.zero_grad()

            # optimization function
            train_cf_loss = [model_fit(train_cf_pred[0], train_label[:, 0]),
                            model_fit(train_cf_pred[1], train_label[:, 1])]

            train_loss = [model_fit(train_pred[0], train_label[:, 0]),
                          model_fit(train_pred[1], train_label[:, 1])]

            t_1_2_cf_loss = [model_fit(t_1_2_cf_pred[0], train_label[:, 0]),
                             model_fit(t_1_2_cf_pred[1], train_label[:, 1])]

            if torch.isnan(train_cf_loss[0]).any():
                print("Epoch{}: batch_index:{} train_cf_loss{} is nan".format(epoch, b, 0))
            if torch.isnan(train_cf_loss[1]).any():
                print("Epoch{}: batch_index:{} train_cf_loss{} is nan".format(epoch, b, 1))

            if torch.isnan(train_loss[0]).any():
                print("Epoch{}: batch_index:{} train_loss{} is nan".format(epoch, b, 0))
            if torch.isnan(train_loss[1]).any():
                print("Epoch{}: batch_index:{} train_loss{} is nan".format(epoch, b, 1))

            if torch.isnan(t_1_2_cf_loss[0]).any():
                print("Epoch{}: batch_index:{} t_1_2_cf_loss{} is nan".format(epoch, b, 0))
            if torch.isnan(t_1_2_cf_loss[1]).any():
                print("Epoch{}: batch_index:{} t_1_2_cf_loss{} is nan".format(epoch, b, 1))

            temp_loss, cls_1, cls_2 = [0, 0], [0, 0], [0, 0]

            Granger_1_to_2 = torch.exp((t_1_2_cf_loss[1] - train_cf_loss[1]) / t_1_2_cf_loss[1])
            Granger_2_to_1 = torch.exp((t_1_2_cf_loss[0] - train_cf_loss[0]) / t_1_2_cf_loss[0])

            for i in range(2):
                cls_1[i] = train_loss[i] / (train_loss[i] + train_cf_loss[i])
                if i == 0:
                    granger_cls = (1 / Granger_2_to_1) * train_loss[1]
                elif i == 1:
                    granger_cls = (1 / Granger_1_to_2) * train_loss[0]

                cls_2[i] = train_loss[i] / (train_loss[i] + granger_cls)
                temp_loss[i] = lambda_lossterm_weight[i][0, epoch] * train_loss[i] +lambda_lossterm_weight[i][1, epoch] * cls_1[i] + lambda_lossterm_weight[i][2, epoch] * cls_2[i]
                avg_loss_term[i][epoch, 0] = train_loss[i].item()
                avg_loss_term[i][epoch, 1] = cls_1[i].item()
                avg_loss_term[i][epoch, 2] = cls_2[i].item()

            loss = sum([lambda_weight[i, epoch] * temp_loss[i] for i in range(2)])

            loss.backward()
            optimizer.step()

            output1 = train_pred[0].max(1, keepdim=True)[1]  # [:, 0]
            output2 = train_pred[1].max(1, keepdim=True)[1]  # [:, 1]
            correct1_train += output1.eq(train_label[:, 0].view_as(output1)).sum().item()
            correct2_train += output2.eq(train_label[:, 1].view_as(output2)).sum().item()

            cost[0] = temp_loss[0].item()
            cost[2] = temp_loss[1].item()
            avg_cost[epoch, :4] += cost[:4] / train_batch

        avg_cost[epoch, 1] = 1.0 * correct1_train / len(train_loader.dataset)
        avg_cost[epoch, 3] = 1.0 * correct2_train / len(train_loader.dataset)

        # test step
        SegNet_MTAN.eval()

        test_dataset = iter(test_loader)
        correct1_test, correct2_test = 0, 0
        for b in range(test_batch):
            test_data, test_label = test_dataset.next()
            test_data, test_label = test_data.to(device), test_label.to(device)

            with torch.no_grad():
                test_pred = SegNet_MTAN(test_data, training=False)
                test_loss = [model_fit(test_pred[0], test_label[:, 0]),
                             model_fit(test_pred[1], test_label[:, 1])]

            output1 = test_pred[0].max(1, keepdim=True)[1]
            output2 = test_pred[1].max(1, keepdim=True)[1]
            correct1_test += output1.eq(test_label[:, 0].view_as(output1)).sum().item()
            correct2_test += output2.eq(test_label[:, 1].view_as(output2)).sum().item()

            cost[4] = test_loss[0].item()
            cost[6] = test_loss[1].item()
            avg_cost[epoch, 4:8] += cost[4:8] / train_batch

        avg_cost[epoch, 5] = 1.0 * correct1_test / len(test_loader.dataset)
        avg_cost[epoch, 7] = 1.0 * correct2_test / len(test_loader.dataset)

        scheduler.step()

        end_time = time.time()
        print(
            'Epoch: {:04d} | TRAIN: {:.4f} {:.4f} | {:.4f} {:.4f} || TEST: {:.4f} {:.4f} | {:.4f} {:.4f} | TIME: {:.4f}'
                .format(epoch, avg_cost[epoch, 0], avg_cost[epoch, 1], avg_cost[epoch, 2], avg_cost[epoch, 3],
                        avg_cost[epoch, 4], avg_cost[epoch, 5], avg_cost[epoch, 6], avg_cost[epoch, 7],
                        end_time - start_time))



def main():
    parser = argparse.ArgumentParser(description='Multi-task: Attention Network')
    parser.add_argument('--method', default='mtan', type=str, help='which optimization algorithm to use')
    parser.add_argument('--weight', default='equal', type=str, help='multi-task weighting: equal, uncert, dwa')
    parser.add_argument('--dataroot', default='doublemnist', type=str, help='dataset root')
    parser.add_argument('--temp', default=2.0, type=float, help='temperature for DWA (must be positive)')
    parser.add_argument('--alpha', default=0.5, type=float, help='the alpha')
    parser.add_argument('--lr', default=1e-4, type=float, help='the learning rate')
    parser.add_argument('--seed', default=0, type=int, help='control seed')
    parser.add_argument('--apply_augmentation', action='store_true',
                        help='toggle to apply data augmentation on doublemnist')
    opt = parser.parse_args()

    # control seed
    torch.backends.cudnn.enabled = False
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    random.seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

    print('LOSS FORMAT: LEFT_LOSS LEFT_ACC | RIGHT_LOSS RIGHT_ACC |')

    # load dataset

    train_txt_path = os.path.join("/code/CFS_DoubleMnist/double_mnist", "train.txt")
    train_dir = os.path.join("/code/CFS_DoubleMnist/double_mnist", "train")

    test_txt_path = os.path.join("/code/CFS_DoubleMnist/double_mnist", "test.txt")
    test_dir = os.path.join("/code/CFS_DoubleMnist/double_mnist", "test")
    gen_txt(train_txt_path, train_dir)
    gen_txt(test_txt_path, test_dir)

    transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
    train_dataset = MyDataset(train_txt_path, transform=transform)
    test_dataset = MyDataset(test_txt_path, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=True, num_workers=2)


    # define model, optimiser and scheduler
    SegNet_MTAN = SegNet_Gumbel_Softmax(k, patch_size, tau, device).to(device)
    optimizer = optim.Adam(SegNet_MTAN.parameters(), lr=opt.lr)
    start_epoch = 0
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    print('start epoch:', start_epoch)

    cfs_cf_contrast(train_loader, test_loader, SegNet_MTAN, optimizer, scheduler, num_epochs, start_epoch)


if __name__ == '__main__':
    main()
