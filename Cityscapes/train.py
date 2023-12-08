import gc
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
import torch.utils.data.sampler as sampler
from cfs_utils import *
from model_segnet_selector_mt import *


# training cfs_models
def cfs_cf_contrast(train_loader, test_loader, SegNet_MTAN, optimizer, scheduler, num_epochs, start_epoch):
    T = opt.temp
    alpha = opt.alpha
    softmax = nn.Softmax(dim=1)

    # variable for keeping track of best ph_acc across different iterations
    avg_cost = np.zeros([num_epochs, 12], dtype=np.float32)
    avg_loss_term = [np.zeros([num_epochs, 3], dtype=np.float32) for _ in range(2)]
    lambda_weight = np.ones([2, num_epochs])
    lambda_lossterm_weight = [np.ones([3, num_epochs]) for _ in range(2)]
    w_loss_term = [[0] * 3 for _ in range(2)]
    train_batch, test_batch = len(train_loader), len(test_loader)

    # training and testing loop
    for epoch in range(start_epoch, num_epochs):
        start_time = time.time()
        conf_mat = ConfMatrix(SegNet_MTAN.class_nb)
        cost = np.zeros(12, dtype=np.float32)

        # compute weights
        if epoch == start_epoch or epoch == start_epoch + 1:
            lambda_weight[:, epoch] = 1.0
            lambda_lossterm_weight[:][:,epoch] = 1.0
        else:
            w_1 = avg_cost[epoch - 1, 0] / avg_cost[epoch - 2, 0]
            w_2 = avg_cost[epoch - 1, 3] / avg_cost[epoch - 2, 3]
            for i in range(2):
                for j in range(3):
                    w_loss_term[i][j] = avg_loss_term[i][epoch - 1, j] / avg_loss_term[i][epoch - 2, j]

            for i in range(2):
                for j in range(3):
                    lambda_lossterm_weight[i][j,epoch] = 3 * np.exp(w_loss_term[i][j] / T) / sum(np.exp(w_loss_term[i][l] / T) for l in range(3))
            lambda_weight[0, epoch] = 2 * np.exp(w_1 / T) / (np.exp(w_1 / T) + np.exp(w_2 / T))
            lambda_weight[1, epoch] = 2 * np.exp(w_2 / T) / (np.exp(w_1 / T) + np.exp(w_2 / T))

        SegNet_MTAN.train()
        train_dataset = iter(train_loader)
        conf_mat = ConfMatrix(SegNet_MTAN.class_nb)
        # training step
        for b in range(train_batch):
            train_data, train_label, train_depth = train_dataset.next()
            train_data, train_label = train_data.to(device), train_label.long().to(device)
            train_depth = train_depth.to(device)
            batch_size = train_data.size(0)

            train_pred, train_cf_pred, t_1_2_cf_pred = SegNet_MTAN(train_data, training=True)

            # zero the parameter gradients
            optimizer.zero_grad()

            # optimization function
            train_cf_loss = [model_fit(train_cf_pred[0], train_label, 'semantic'),
                             model_fit(train_cf_pred[1], train_depth, 'depth')]

            train_loss = [model_fit(train_pred[0], train_label, 'semantic'),
                          model_fit(train_pred[1], train_depth, 'depth')]

            t_1_2_cf_loss = [model_fit(t_1_2_cf_pred[0], train_label, 'semantic'),
                             model_fit(t_1_2_cf_pred[1], train_depth, 'depth')]

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
                temp_loss[i] = lambda_lossterm_weight[i][0,epoch] * train_loss[i] + lambda_lossterm_weight[i][1,epoch] * cls_1[i] + lambda_lossterm_weight[i][2,epoch] * cls_2[i]
                avg_loss_term[i][epoch, 0] = train_loss[i].item()
                avg_loss_term[i][epoch, 1] = cls_1[i].item()
                avg_loss_term[i][epoch, 2] = cls_2[i].item()

            loss = sum([lambda_weight[i, epoch] * temp_loss[i] for i in range(2)])

            
            loss.backward()
            optimizer.step()

            # accumulate label prediction for every pixel in training images
            conf_mat.update(train_pred[0].argmax(1).flatten(), train_label.flatten())

            cost[0] = temp_loss[0].item()
            cost[3] = temp_loss[1].item()
            cost[4], cost[5] = depth_error(train_pred[1], train_depth)
            avg_cost[epoch, :6] += cost[:6] / train_batch

            # compute mIoU and acc
        avg_cost[epoch, 1:3] = conf_mat.get_metrics()

        # test step
        SegNet_MTAN.eval()
        conf_mat = ConfMatrix(SegNet_MTAN.class_nb)
        test_dataset = iter(test_loader)
        for b in range(test_batch):
            test_data, test_label, test_depth = test_dataset.next()
            test_data, test_label = test_data.to(device), test_label.long().to(device)
            test_depth = test_depth.to(device)

            with torch.no_grad():
                test_pred = SegNet_MTAN(test_data, training=False)
                test_loss = [model_fit(test_pred[0], test_label, 'semantic'),
                             model_fit(test_pred[1], test_depth, 'depth')]

            conf_mat.update(test_pred[0].argmax(1).flatten(), test_label.flatten())

            cost[6] = test_loss[0].item()
            cost[9] = test_loss[1].item()
            cost[10], cost[11] = depth_error(test_pred[1], test_depth)
            avg_cost[epoch, 6:] += cost[6:] / test_batch

        # compute mIoU and acc
        avg_cost[epoch, 7:9] = conf_mat.get_metrics()

        scheduler.step()

        end_time = time.time()
        print(
            'pretrain cfs Epoch: {:04d} |''Train: {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} ||TEST: {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} | {:.4f}'.format(
                epoch, avg_cost[epoch, 0], avg_cost[epoch, 1], avg_cost[epoch, 2],
                avg_cost[epoch, 3], avg_cost[epoch, 4],
                avg_cost[epoch, 5], avg_cost[epoch, 6], avg_cost[epoch, 7], avg_cost[epoch, 8],
                avg_cost[epoch, 9], avg_cost[epoch, 10],
                avg_cost[epoch, 11], end_time - start_time))



def main():
    parser = argparse.ArgumentParser(description='Multi-task: Attention Network')
    parser.add_argument('--method', default='mtan', type=str, help='which optimization algorithm to use')
    parser.add_argument('--weight', default='equal', type=str, help='multi-task weighting: equal, uncert, dwa')
    parser.add_argument('--dataroot', default='cityscapes', type=str, help='dataset root')
    parser.add_argument('--temp', default=2.0, type=float, help='temperature for DWA (must be positive)')
    parser.add_argument('--alpha', default=0.5, type=float, help='the alpha')
    parser.add_argument('--lr', default=1e-4, type=float, help='the learning rate')
    parser.add_argument('--seed', default=0, type=int, help='control seed')
    parser.add_argument('--apply_augmentation', action='store_true',
                        help='toggle to apply data augmentation on cityscapes')
    opt = parser.parse_args()

    # control seed
    torch.backends.cudnn.enabled = False
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    random.seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('LOSS FORMAT: SEMANTIC_LOSS MEAN_IOU PIX_ACC | DEPTH_LOSS ABS_ERR REL_ERR |')

    # define dataset
    dataset_path = opt.dataroot
    if opt.apply_augmentation:
        train_set = CityScapes(root=dataset_path, train=True, augmentation=True)
        print('Applying data augmentation.')
    else:
        train_set = CityScapes(root=dataset_path, train=True)
        print('Standard training strategy without data augmentation.')

    test_set = CityScapes(root=dataset_path, train=False)

    batch_size = 4
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=batch_size,
        shuffle=False)


    # define model, optimiser and scheduler
    SegNet_MTAN = SegNet_Gumbel_Softmax(k, patch_size, tau, device).to(device)
    optimizer = optim.Adam(SegNet_MTAN.parameters(), lr=opt.lr)
    start_epoch = 0
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    num_epochs = 200

    print('Parameter Space: ABS: {:.1f}, REL: {:.4f}'.format(count_parameters(SegNet_MTAN),
                                                             count_parameters(SegNet_MTAN) / 24981069))
    print('start epoch:', start_epoch)

    cfs_cf_contrast(train_loader, test_loader, SegNet_MTAN, optimizer, scheduler, num_epochs, start_epoch)
    
if __name__ == '__main__':
    main()
