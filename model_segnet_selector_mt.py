import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
import torch.utils.data.sampler as sampler

from create_dataset import *
# from utils import *
from cfs_utils import *

parser = argparse.ArgumentParser(description='Multi-task: Attention Network')
parser.add_argument('--method', default='mtan', type=str, help='which optimization algorithm to use')
parser.add_argument('--weight', default='equal', type=str, help='multi-task weighting: equal, uncert, dwa')
parser.add_argument('--dataroot', default='doublemnist', type=str, help='dataset root')
parser.add_argument('--temp', default=2.0, type=float, help='temperature for DWA (must be positive)')
parser.add_argument('--alpha', default=0.5, type=float, help='the alpha')
parser.add_argument('--lr', default=1e-4, type=float, help='the learning rate')
parser.add_argument('--seed', default=0, type=int, help='control seed')
parser.add_argument('--apply_augmentation', action='store_true', help='toggle to apply data augmentation on doublemnist')
opt = parser.parse_args()


class SegNet_Selector(nn.Module):
    def __init__(self):
        super(SegNet_Selector, self).__init__()
        # initialise network parameters
        filter = [8, 16, 32, 64, 64]
        self.class_nb = 10
        self.batch_size = 256

        # define encoder decoder layers
        self.encoder_block = nn.ModuleList([self.conv_layer([1, filter[0]])])
        self.decoder_block = nn.ModuleList([self.conv_layer([filter[0], filter[0]])])
        for i in range(4):
            self.encoder_block.append(self.conv_layer([filter[i], filter[i + 1]]))
            self.decoder_block.append(self.conv_layer([filter[i + 1], filter[i]]))

        # define convolution layer
        self.conv_block_enc = nn.ModuleList([self.conv_layer([filter[0], filter[0]])])
        self.conv_block_dec = nn.ModuleList([self.conv_layer([filter[0], filter[0]])])
        for i in range(4):
            if i == 0:
                self.conv_block_enc.append(self.conv_layer([filter[i + 1], filter[i + 1]]))
                self.conv_block_dec.append(self.conv_layer([filter[i], filter[i]]))
            else:
                self.conv_block_enc.append(nn.Sequential(self.conv_layer([filter[i + 1], filter[i + 1]]),
                                                         self.conv_layer([filter[i + 1], filter[i + 1]])))
                self.conv_block_dec.append(nn.Sequential(self.conv_layer([filter[i], filter[i]]),
                                                         self.conv_layer([filter[i], filter[i]])))

        # define task attention layers
        self.encoder_att = nn.ModuleList([nn.ModuleList([self.att_layer([filter[0], filter[0], filter[0]])])])
        self.decoder_att = nn.ModuleList([nn.ModuleList([self.att_layer([2 * filter[0], filter[0], filter[0]])])])
        self.encoder_block_att = nn.ModuleList([self.conv_layer([filter[0], filter[1]])])
        self.decoder_block_att = nn.ModuleList([self.conv_layer([filter[0], filter[0]])])

        for j in range(2):
            if j < 1:
                self.encoder_att.append(nn.ModuleList([self.att_layer([filter[0], filter[0], filter[0]])]))
                self.decoder_att.append(nn.ModuleList([self.att_layer([2 * filter[0], filter[0], filter[0]])]))
            for i in range(4):
                self.encoder_att[j].append(self.att_layer([2 * filter[i + 1], filter[i + 1], filter[i + 1]]))
                self.decoder_att[j].append(self.att_layer([filter[i + 1] + filter[i], filter[i], filter[i]]))

        for i in range(4):
            if i < 3:
                self.encoder_block_att.append(self.conv_layer([filter[i + 1], filter[i + 2]]))
                self.decoder_block_att.append(self.conv_layer([filter[i + 1], filter[i]]))
            else:
                self.encoder_block_att.append(self.conv_layer([filter[i + 1], filter[i + 1]]))
                self.decoder_block_att.append(self.conv_layer([filter[i + 1], filter[i + 1]]))

        self.pred_task1 = self.conv_layer([filter[0], 1], pred=True)  # self.class_nb
        self.pred_task2 = self.conv_layer([filter[0], 1], pred=True)
        # self.pred_task3 = self.conv_layer([filter[0], 3], pred=True)

        # define pooling and unpooling functions
        self.down_sampling = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.up_sampling = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(64 * 64, self.class_nb)
        self.logsigma = nn.Parameter(torch.FloatTensor([-0.5, -0.5]))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def shared_modules(self):
        return [self.encoder_block, self.decoder_block,
                self.conv_block_enc, self.conv_block_dec,
                self.encoder_block_att, self.decoder_block_att,
                self.down_sampling, self.up_sampling]

    def zero_grad_shared_modules(self):
        for mm in self.shared_modules():
            mm.zero_grad()

    def conv_layer(self, channel, pred=False):
        if not pred:
            conv_block = nn.Sequential(
                nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=3, padding=1),
                nn.BatchNorm2d(num_features=channel[1]),
                nn.ReLU(inplace=True),
            )
        else:
            conv_block = nn.Sequential(
                nn.Conv2d(in_channels=channel[0], out_channels=channel[0], kernel_size=3, padding=1),
                nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=1, padding=0),
            )
        return conv_block

    def out_logits_conv_layer(self, channel):
        conv_block = nn.Sequential(
            nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(num_features=channel[1]),
            nn.ReLU(inplace=True),
            # nn.Conv2d(in_channels=channel[1], out_channels=1, kernel_size=1),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.BatchNorm2d(num_features=channel[1]),
            # nn.ReLU(inplace=True),
        )
        return conv_block

    def att_layer(self, channel):
        att_block = nn.Sequential(
            nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=1, padding=0),
            nn.BatchNorm2d(channel[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=channel[1], out_channels=channel[2], kernel_size=1, padding=0),
            nn.BatchNorm2d(channel[2]),
            nn.Sigmoid(),
        )
        return att_block

    def forward(self, x):
        g_encoder, g_decoder, g_maxpool, g_upsampl, indices = ([0] * 5 for _ in range(5))
        for i in range(5):
            g_encoder[i], g_decoder[-i - 1] = ([0] * 2 for _ in range(2))

        # define attention list for tasks
        atten_encoder, atten_decoder = ([0] * 2 for _ in range(2))
        for i in range(2):
            atten_encoder[i], atten_decoder[i] = ([0] * 5 for _ in range(2))
        for i in range(2):
            for j in range(5):
                atten_encoder[i][j], atten_decoder[i][j] = ([0] * 3 for _ in range(2))

        # define global shared network
        for i in range(5):
            if i == 0:
                g_encoder[i][0] = self.encoder_block[i](x)
                g_encoder[i][1] = self.conv_block_enc[i](g_encoder[i][0])
                g_maxpool[i], indices[i] = self.down_sampling(g_encoder[i][1])
            else:
                g_encoder[i][0] = self.encoder_block[i](g_maxpool[i - 1])
                g_encoder[i][1] = self.conv_block_enc[i](g_encoder[i][0])
                g_maxpool[i], indices[i] = self.down_sampling(g_encoder[i][1])

        for i in range(5):
            if i == 0:
                g_upsampl[i] = self.up_sampling(g_maxpool[-1], indices[-i - 1])
                g_decoder[i][0] = self.decoder_block[-i - 1](g_upsampl[i])
                g_decoder[i][1] = self.conv_block_dec[-i - 1](g_decoder[i][0])
            else:
                g_upsampl[i] = self.up_sampling(g_decoder[i - 1][-1], indices[-i - 1])
                g_decoder[i][0] = self.decoder_block[-i - 1](g_upsampl[i])
                g_decoder[i][1] = self.conv_block_dec[-i - 1](g_decoder[i][0])

        # define task dependent attention module
        for i in range(2):
            for j in range(5):
                if j == 0:
                    atten_encoder[i][j][0] = self.encoder_att[i][j](g_encoder[j][0])
                    atten_encoder[i][j][1] = (atten_encoder[i][j][0]) * g_encoder[j][1]
                    atten_encoder[i][j][2] = self.encoder_block_att[j](atten_encoder[i][j][1])
                    atten_encoder[i][j][2] = F.max_pool2d(atten_encoder[i][j][2], kernel_size=2, stride=2)
                else:
                    atten_encoder[i][j][0] = self.encoder_att[i][j](
                        torch.cat((g_encoder[j][0], atten_encoder[i][j - 1][2]), dim=1))
                    atten_encoder[i][j][1] = (atten_encoder[i][j][0]) * g_encoder[j][1]
                    atten_encoder[i][j][2] = self.encoder_block_att[j](atten_encoder[i][j][1])
                    atten_encoder[i][j][2] = F.max_pool2d(atten_encoder[i][j][2], kernel_size=2, stride=2)

            for j in range(5):
                if j == 0:
                    atten_decoder[i][j][0] = F.interpolate(atten_encoder[i][-1][-1], scale_factor=2, mode='bilinear',
                                                           align_corners=True)
                    atten_decoder[i][j][0] = self.decoder_block_att[-j - 1](atten_decoder[i][j][0])
                    atten_decoder[i][j][1] = self.decoder_att[i][-j - 1](
                        torch.cat((g_upsampl[j], atten_decoder[i][j][0]), dim=1))
                    atten_decoder[i][j][2] = (atten_decoder[i][j][1]) * g_decoder[j][-1]
                else:
                    atten_decoder[i][j][0] = F.interpolate(atten_decoder[i][j - 1][2], scale_factor=2, mode='bilinear',
                                                           align_corners=True)
                    atten_decoder[i][j][0] = self.decoder_block_att[-j - 1](atten_decoder[i][j][0])
                    atten_decoder[i][j][1] = self.decoder_att[i][-j - 1](
                        torch.cat((g_upsampl[j], atten_decoder[i][j][0]), dim=1))
                    atten_decoder[i][j][2] = (atten_decoder[i][j][1]) * g_decoder[j][-1]

        # define task prediction layers
        t1_pred = self.pred_task1(atten_decoder[0][-1][-1])
        t2_pred = self.pred_task2(atten_decoder[1][-1][-1])

        return [t1_pred, t2_pred]



class SegNet(nn.Module):
    def __init__(self):
        super(SegNet, self).__init__()
        # initialise network parameters
        filter = [8, 16, 32, 64, 64]
        self.class_nb = 10
        self.batch_size = 256

        # define encoder decoder layers
        self.encoder_block = nn.ModuleList([self.conv_layer([1, filter[0]])])
        self.decoder_block = nn.ModuleList([self.conv_layer([filter[0], filter[0]])])
        for i in range(4):
            self.encoder_block.append(self.conv_layer([filter[i], filter[i + 1]]))
            self.decoder_block.append(self.conv_layer([filter[i + 1], filter[i]]))

        # define convolution layer
        self.conv_block_enc = nn.ModuleList([self.conv_layer([filter[0], filter[0]])])
        self.conv_block_dec = nn.ModuleList([self.conv_layer([filter[0], filter[0]])])
        for i in range(4):
            if i == 0:
                self.conv_block_enc.append(self.conv_layer([filter[i + 1], filter[i + 1]]))
                self.conv_block_dec.append(self.conv_layer([filter[i], filter[i]]))
            else:
                self.conv_block_enc.append(nn.Sequential(self.conv_layer([filter[i + 1], filter[i + 1]]),
                                                         self.conv_layer([filter[i + 1], filter[i + 1]])))
                self.conv_block_dec.append(nn.Sequential(self.conv_layer([filter[i], filter[i]]),
                                                         self.conv_layer([filter[i], filter[i]])))

        # define task attention layers
        self.encoder_att = nn.ModuleList([nn.ModuleList([self.att_layer([filter[0], filter[0], filter[0]])])])
        self.decoder_att = nn.ModuleList([nn.ModuleList([self.att_layer([2 * filter[0], filter[0], filter[0]])])])
        self.encoder_block_att = nn.ModuleList([self.conv_layer([filter[0], filter[1]])])
        self.decoder_block_att = nn.ModuleList([self.conv_layer([filter[0], filter[0]])])

        for j in range(2):
            if j < 1:
                self.encoder_att.append(nn.ModuleList([self.att_layer([filter[0], filter[0], filter[0]])]))
                self.decoder_att.append(nn.ModuleList([self.att_layer([2 * filter[0], filter[0], filter[0]])]))
            for i in range(4):
                self.encoder_att[j].append(self.att_layer([2 * filter[i + 1], filter[i + 1], filter[i + 1]]))
                self.decoder_att[j].append(self.att_layer([filter[i + 1] + filter[i], filter[i], filter[i]]))

        for i in range(4):
            if i < 3:
                self.encoder_block_att.append(self.conv_layer([filter[i + 1], filter[i + 2]]))
                self.decoder_block_att.append(self.conv_layer([filter[i + 1], filter[i]]))
            else:
                self.encoder_block_att.append(self.conv_layer([filter[i + 1], filter[i + 1]]))
                self.decoder_block_att.append(self.conv_layer([filter[i + 1], filter[i + 1]]))

        self.pred_task1 = self.conv_layer([filter[0], 1], pred=True)
        self.pred_task2 = self.conv_layer([filter[0], 1], pred=True)
        # self.pred_task3 = self.conv_layer([filter[0], 3], pred=True)

        # define pooling and unpooling functions
        self.down_sampling = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.up_sampling = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(64 * 64, self.class_nb)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def conv_layer(self, channel, pred=False):
        if not pred:
            conv_block = nn.Sequential(
                nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=3, padding=1),
                nn.BatchNorm2d(num_features=channel[1]),
                nn.ReLU(inplace=True),
            )
        else:
            conv_block = nn.Sequential(
                nn.Conv2d(in_channels=channel[0], out_channels=channel[0], kernel_size=3, padding=1),
                nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=1, padding=0),
            )
        return conv_block

    def att_layer(self, channel):
        att_block = nn.Sequential(
            nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=1, padding=0),
            nn.BatchNorm2d(channel[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=channel[1], out_channels=channel[2], kernel_size=1, padding=0),
            nn.BatchNorm2d(channel[2]),
            nn.Sigmoid(),
        )
        return att_block

    def forward(self, x):
        self.batch_size = x.size(0)
        g_encoder, g_decoder, g_maxpool, g_upsampl, indices = ([0] * 5 for _ in range(5))
        for i in range(5):
            g_encoder[i], g_decoder[-i - 1] = ([0] * 2 for _ in range(2))

        # define attention list for tasks
        atten_encoder, atten_decoder = ([0] * 2 for _ in range(2))
        for i in range(2):
            atten_encoder[i], atten_decoder[i] = ([0] * 5 for _ in range(2))
        for i in range(2):
            for j in range(5):
                atten_encoder[i][j], atten_decoder[i][j] = ([0] * 3 for _ in range(2))

        # define global shared network
        for i in range(5):
            if i == 0:
                g_encoder[i][0] = self.encoder_block[i](x)
                g_encoder[i][1] = self.conv_block_enc[i](g_encoder[i][0])
                g_maxpool[i], indices[i] = self.down_sampling(g_encoder[i][1])
            else:
                g_encoder[i][0] = self.encoder_block[i](g_maxpool[i - 1])
                g_encoder[i][1] = self.conv_block_enc[i](g_encoder[i][0])
                g_maxpool[i], indices[i] = self.down_sampling(g_encoder[i][1])

        for i in range(5):
            if i == 0:
                g_upsampl[i] = self.up_sampling(g_maxpool[-1], indices[-i - 1])
                g_decoder[i][0] = self.decoder_block[-i - 1](g_upsampl[i])
                g_decoder[i][1] = self.conv_block_dec[-i - 1](g_decoder[i][0])
            else:
                g_upsampl[i] = self.up_sampling(g_decoder[i - 1][-1], indices[-i - 1])
                g_decoder[i][0] = self.decoder_block[-i - 1](g_upsampl[i])
                g_decoder[i][1] = self.conv_block_dec[-i - 1](g_decoder[i][0])

        # define task dependent attention module
        for i in range(2):
            for j in range(5):
                if j == 0:
                    atten_encoder[i][j][0] = self.encoder_att[i][j](g_encoder[j][0])
                    atten_encoder[i][j][1] = (atten_encoder[i][j][0]) * g_encoder[j][1]
                    atten_encoder[i][j][2] = self.encoder_block_att[j](atten_encoder[i][j][1])
                    atten_encoder[i][j][2] = F.max_pool2d(atten_encoder[i][j][2], kernel_size=2, stride=2)
                else:
                    atten_encoder[i][j][0] = self.encoder_att[i][j](
                        torch.cat((g_encoder[j][0], atten_encoder[i][j - 1][2]), dim=1))
                    atten_encoder[i][j][1] = (atten_encoder[i][j][0]) * g_encoder[j][1]
                    atten_encoder[i][j][2] = self.encoder_block_att[j](atten_encoder[i][j][1])
                    atten_encoder[i][j][2] = F.max_pool2d(atten_encoder[i][j][2], kernel_size=2, stride=2)

            for j in range(5):
                if j == 0:
                    atten_decoder[i][j][0] = F.interpolate(atten_encoder[i][-1][-1], scale_factor=2, mode='bilinear',
                                                           align_corners=True)
                    atten_decoder[i][j][0] = self.decoder_block_att[-j - 1](atten_decoder[i][j][0])
                    atten_decoder[i][j][1] = self.decoder_att[i][-j - 1](
                        torch.cat((g_upsampl[j], atten_decoder[i][j][0]), dim=1))
                    atten_decoder[i][j][2] = (atten_decoder[i][j][1]) * g_decoder[j][-1]
                else:
                    atten_decoder[i][j][0] = F.interpolate(atten_decoder[i][j - 1][2], scale_factor=2, mode='bilinear',
                                                           align_corners=True)
                    atten_decoder[i][j][0] = self.decoder_block_att[-j - 1](atten_decoder[i][j][0])
                    atten_decoder[i][j][1] = self.decoder_att[i][-j - 1](
                        torch.cat((g_upsampl[j], atten_decoder[i][j][0]), dim=1))
                    atten_decoder[i][j][2] = (atten_decoder[i][j][1]) * g_decoder[j][-1]

        # define task prediction layers
        t1_pred = F.softmax(self.fc(self.pred_task1(atten_decoder[0][-1][-1]).view(self.batch_size, -1)), dim=1)
        t2_pred = F.softmax(self.fc(self.pred_task2(atten_decoder[1][-1][-1]).view(self.batch_size, -1)), dim=1)
        # t3_pred = self.pred_task3(atten_decoder[2][-1][-1])
        # t3_pred = t3_pred / torch.norm(t3_pred, p=2, dim=1, keepdim=True)

        return [t1_pred, t2_pred]

class SegNet_Gumbel_Softmax(nn.Module):
    def __init__(self, k, patch_size, Tau_coe, device):
        super(SegNet_Gumbel_Softmax, self).__init__()
        # initialise network parameters
        filter = [8, 16, 32, 64, 64]
        self.class_nb = 10
        self.batch_size = 256
        self.k = k
        self.patch_size = patch_size
        self.Tau_coe = Tau_coe
        self.device = device
        self.noise_factor = 0.3

        # define encoder decoder layers
        self.encoder_block = nn.ModuleList([self.conv_layer([1, filter[0]])])
        self.decoder_block = nn.ModuleList([self.conv_layer([filter[0], filter[0]])])
        for i in range(4):
            self.encoder_block.append(self.conv_layer([filter[i], filter[i + 1]]))
            self.decoder_block.append(self.conv_layer([filter[i + 1], filter[i]]))

        # define convolution layer
        self.conv_block_enc = nn.ModuleList([self.conv_layer([filter[0], filter[0]])])
        self.conv_block_dec = nn.ModuleList([self.conv_layer([filter[0], filter[0]])])
        for i in range(4):
            if i == 0:
                self.conv_block_enc.append(self.conv_layer([filter[i + 1], filter[i + 1]]))
                self.conv_block_dec.append(self.conv_layer([filter[i], filter[i]]))
            else:
                self.conv_block_enc.append(nn.Sequential(self.conv_layer([filter[i + 1], filter[i + 1]]),
                                                         self.conv_layer([filter[i + 1], filter[i + 1]])))
                self.conv_block_dec.append(nn.Sequential(self.conv_layer([filter[i], filter[i]]),
                                                         self.conv_layer([filter[i], filter[i]])))

        # define task attention layers
        self.encoder_att = nn.ModuleList([nn.ModuleList([self.att_layer([filter[0], filter[0], filter[0]])])])
        self.decoder_att = nn.ModuleList([nn.ModuleList([self.att_layer([2 * filter[0], filter[0], filter[0]])])])
        self.encoder_block_att = nn.ModuleList([self.conv_layer([filter[0], filter[1]])])
        self.decoder_block_att = nn.ModuleList([self.conv_layer([filter[0], filter[0]])])

        for j in range(2):
            if j < 1:
                self.encoder_att.append(nn.ModuleList([self.att_layer([filter[0], filter[0], filter[0]])]))
                self.decoder_att.append(nn.ModuleList([self.att_layer([2 * filter[0], filter[0], filter[0]])]))
            for i in range(4):
                self.encoder_att[j].append(self.att_layer([2 * filter[i + 1], filter[i + 1], filter[i + 1]]))
                self.decoder_att[j].append(self.att_layer([filter[i + 1] + filter[i], filter[i], filter[i]]))

        for i in range(4):
            if i < 3:
                self.encoder_block_att.append(self.conv_layer([filter[i + 1], filter[i + 2]]))
                self.decoder_block_att.append(self.conv_layer([filter[i + 1], filter[i]]))
            else:
                self.encoder_block_att.append(self.conv_layer([filter[i + 1], filter[i + 1]]))
                self.decoder_block_att.append(self.conv_layer([filter[i + 1], filter[i + 1]]))

        self.pred_task_1 = self.conv_layer([filter[0], 1], pred=True)
        self.pred_task_2 = self.conv_layer([filter[0], 1], pred=True)
        self.pred_task_logits = self.out_logits_conv_layer([filter[0], 1])

        # define pooling and unpooling functions
        self.down_sampling = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.up_sampling = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.fc_1 = nn.Linear(64 * 64, self.class_nb)
        self.fc_2 = nn.Linear(64 * 64, self.class_nb)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def shared_modules(self):
        return [self.encoder_block, self.decoder_block,
                self.conv_block_enc, self.conv_block_dec,
                self.encoder_block_att, self.decoder_block_att,
                self.down_sampling, self.up_sampling]

    def zero_grad_shared_modules(self):
        for mm in self.shared_modules():
            mm.zero_grad()

    def conv_layer(self, channel, pred=False):
        if not pred:
            conv_block = nn.Sequential(
                nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=3, padding=1),
                nn.BatchNorm2d(num_features=channel[1]),
                nn.ReLU(inplace=True),
            )
        else:
            conv_block = nn.Sequential(
                nn.Conv2d(in_channels=channel[0], out_channels=channel[0], kernel_size=3, padding=1),
                nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=1, padding=0),
            )
        return conv_block

    def out_logits_conv_layer(self, channel):
        conv_block = nn.Sequential(
            nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Conv2d(in_channels=channel[1], out_channels=1, kernel_size=1),
            # nn.MaxPool2d(kernel_size=2, stride=2),
        )
        return conv_block

    def att_layer(self, channel):
        att_block = nn.Sequential(
            nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=1, padding=0),
            nn.BatchNorm2d(channel[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=channel[1], out_channels=channel[2], kernel_size=1, padding=0),
            nn.BatchNorm2d(channel[2]),
            nn.Sigmoid(),
        )
        return att_block

    def sample(self, logits):
        # print(logits.size())
        batch_size, channels, d1, d2 = logits.size()
        if self.batch_size != batch_size:
            self.batch_size = batch_size
        dims_s = self.batch_size * channels * d1 * d2
        # upsample_op = nn.Upsample(scale_factor=self.patch_size, mode='nearest')
        K = self.batch_size * self.k * channels
        sample_logits = torch.zeros_like(logits.view(-1, dims_s))  # (-1, bactch * logits0 * logits1)

        logits = torch.reshape(logits, (-1, dims_s))  # (-1, bactch * logits0 * logits1)
        vals, indics = torch.topk(logits, K, dim=-1)
        sample_logits[0, indics[0, :]] = 1
        del vals, indics

        sample_logits = torch.reshape(sample_logits, (self.batch_size, channels, d1, d2))
        # imp = upsample_op(sample_logits)
        del logits
        return sample_logits

    def Sample_gumbel(self, logits):
        d1, d2 = logits.shape[2], logits.shape[3]
        dims = d1 * d2

        if self.training == True:
            softmax = nn.Softmax(dim=-1)
            logits = logits.view(self.batch_size, -1).unsqueeze(1)
            # print("logits.size()", logits.size())
            unif_shape = [self.batch_size, self.k, dims]
            # print("unif_shape", unif_shape)
            uniform = (1 - 0) * torch.rand(unif_shape)

            gumbel = - torch.log(-torch.log(uniform)).to(self.device)
            # print("gumbel.size()", gumbel.size())
            noisy_logits = (gumbel + logits) / self.Tau_coe
            samples = softmax(noisy_logits)
            samples, samples_ids = torch.max(samples, dim=1)
            # print("samples.size()", samples.size())

            del logits, unif_shape, uniform, gumbel, noisy_logits, samples_ids
            samples = torch.reshape(samples, (self.batch_size, d1, d2))
            selected_subset = torch.unsqueeze(samples, dim=1)

            upsample_op = nn.Upsample(scale_factor=self.patch_size, mode='nearest')
            imp = upsample_op(selected_subset)
            # print("imp.size()", imp.size())
            del samples, selected_subset
        else:
            imp = self.sample(logits)
        return imp

    '''
    def gumbel_sample(self, logits):
        batch_size, channels, d1, d2 = logits.size()
        dims = d1 * d2
        gumbel_samples = F.gumbel_softmax(logits, tau=self.Tau_coe, hard=False)
        imp_index = self.sample(gumbel_samples)
        noise_index = torch.ones_like(imp_index) - imp_index

        return imp_index, noise_index
    '''

    def gumbel_sample(self, logits):
        batch_size, channels, d1, d2 = logits.size()
        dims = d1 * d2

        gumbel_samples = F.gumbel_softmax(logits, tau=self.Tau_coe, hard=False)

        for i in range(self.k - 1):
            gumbel_sample_k_time = F.gumbel_softmax(logits, tau=self.Tau_coe, hard=False)
            gumbel_sample_k_times = torch.stack((gumbel_samples, gumbel_sample_k_time), dim=0)
            gumbel_samples, ids = torch.max(gumbel_sample_k_times, dim=0)
            del gumbel_sample_k_time, gumbel_sample_k_times, ids

        return gumbel_samples

    '''
    def Add_noise(self, feas, imp_index, noise_index, noise_factor):
        noisy_cf = feas + torch.randn_like(feas) * noise_index * noise_factor
        # noisy_cf = feas * imp_index + torch.randn_like(feas) * noise_index
        return noisy_cf
    '''
    def Add_noise(self, feas, imp_scores, noise_factor):
        #noisy_cf = feas * imp_scores + torch.randn_like(feas * imp_scores) * noise_factor
        noisy_cf = feas + torch.randn_like(feas) * noise_factor * imp_scores
        return noisy_cf

    def Add_random_noise(self, feas, noise_factor):
        noisy_cf = feas + torch.randn_like(feas) * noise_factor
        return noisy_cf

    def Union_task_gumbel_index(self, task1_index, task2_index):

        batch_size, channels, d1, d2 = task1_index.size()
        dims = batch_size * channels * d1 * d2
        gumbel_index_Union_1_2 = (task1_index + task2_index).view(dims)
        # print("gumbel_index_Union_1_2.size()", gumbel_index_Union_1_2.size())
        gumbel_sample_index_1_2 = torch.zeros_like(gumbel_index_Union_1_2)
        # print("gumbel_sample_index_1_2.size()", gumbel_sample_index_1_2.size())

        nonzero_index = torch.nonzero(gumbel_index_Union_1_2)
        # print("nonzero_index.size()", nonzero_index.size())
        # print("nonzero_index", nonzero_index)
        gumbel_sample_index_1_2[nonzero_index] = 1
        noise_index_1_2 = torch.ones_like(gumbel_sample_index_1_2) - gumbel_sample_index_1_2

        gumbel_sample_index_1_2 = gumbel_sample_index_1_2.view(batch_size, channels, d1, d2)
        noise_index_1_2 = noise_index_1_2.view(batch_size, channels, d1, d2)

        del batch_size, channels, d1, d2, gumbel_index_Union_1_2, nonzero_index

        return gumbel_sample_index_1_2, noise_index_1_2

    def forward(self, x, training=True, random_bu=False):
        self.batch_size = x.size(0)
        self.training = training
        self.random_bu = random_bu
        g_encoder, g_decoder, g_maxpool, g_upsampl, indices = ([0] * 5 for _ in range(5))
        for i in range(5):
            g_encoder[i], g_decoder[-i - 1] = ([0] * 2 for _ in range(2))

        # define attention list for tasks
        atten_encoder, atten_decoder = ([0] * 2 for _ in range(2))
        att_cf_decoder, gumbel_logits = ([0] * 2 for _ in range(2))
        sample_gumbel_index = [0] * 2
        for i in range(2):
            atten_encoder[i], atten_decoder[i] = ([0] * 5 for _ in range(2))
            sample_gumbel_index[i] = [0] * 2
        for i in range(2):
            for j in range(5):
                atten_encoder[i][j], atten_decoder[i][j] = ([0] * 3 for _ in range(2))

        # define global shared network
        for i in range(5):
            if i == 0:
                g_encoder[i][0] = self.encoder_block[i](x)
                g_encoder[i][1] = self.conv_block_enc[i](g_encoder[i][0])
                g_maxpool[i], indices[i] = self.down_sampling(g_encoder[i][1])
            else:
                g_encoder[i][0] = self.encoder_block[i](g_maxpool[i - 1])
                g_encoder[i][1] = self.conv_block_enc[i](g_encoder[i][0])
                g_maxpool[i], indices[i] = self.down_sampling(g_encoder[i][1])

        for i in range(5):
            if i == 0:
                g_upsampl[i] = self.up_sampling(g_maxpool[-1], indices[-i - 1])
                g_decoder[i][0] = self.decoder_block[-i - 1](g_upsampl[i])
                g_decoder[i][1] = self.conv_block_dec[-i - 1](g_decoder[i][0])
            else:
                g_upsampl[i] = self.up_sampling(g_decoder[i - 1][-1], indices[-i - 1])
                g_decoder[i][0] = self.decoder_block[-i - 1](g_upsampl[i])
                g_decoder[i][1] = self.conv_block_dec[-i - 1](g_decoder[i][0])

        # define task dependent attention module
        for i in range(2):
            for j in range(5):
                if j == 0:
                    atten_encoder[i][j][0] = self.encoder_att[i][j](g_encoder[j][0])
                    atten_encoder[i][j][1] = (atten_encoder[i][j][0]) * g_encoder[j][1]
                    atten_encoder[i][j][2] = self.encoder_block_att[j](atten_encoder[i][j][1])
                    atten_encoder[i][j][2] = F.max_pool2d(atten_encoder[i][j][2], kernel_size=2, stride=2)
                else:
                    atten_encoder[i][j][0] = self.encoder_att[i][j](
                        torch.cat((g_encoder[j][0], atten_encoder[i][j - 1][2]), dim=1))
                    atten_encoder[i][j][1] = (atten_encoder[i][j][0]) * g_encoder[j][1]
                    atten_encoder[i][j][2] = self.encoder_block_att[j](atten_encoder[i][j][1])
                    atten_encoder[i][j][2] = F.max_pool2d(atten_encoder[i][j][2], kernel_size=2, stride=2)

            for j in range(5):
                if j == 0:
                    atten_decoder[i][j][0] = F.interpolate(atten_encoder[i][-1][-1], scale_factor=2, mode='bilinear',
                                                           align_corners=True)
                    atten_decoder[i][j][0] = self.decoder_block_att[-j - 1](atten_decoder[i][j][0])
                    atten_decoder[i][j][1] = self.decoder_att[i][-j - 1](
                        torch.cat((g_upsampl[j], atten_decoder[i][j][0]), dim=1))
                    atten_decoder[i][j][2] = (atten_decoder[i][j][1]) * g_decoder[j][-1]
                else:
                    atten_decoder[i][j][0] = F.interpolate(atten_decoder[i][j - 1][2], scale_factor=2, mode='bilinear',
                                                           align_corners=True)
                    atten_decoder[i][j][0] = self.decoder_block_att[-j - 1](atten_decoder[i][j][0])
                    atten_decoder[i][j][1] = self.decoder_att[i][-j - 1](
                        torch.cat((g_upsampl[j], atten_decoder[i][j][0]), dim=1))
                    atten_decoder[i][j][2] = (atten_decoder[i][j][1]) * g_decoder[j][-1]
                    if self.training:
                        if j == 4:
                            gumbel_logits[i] = atten_decoder[i][j][1]
                            sample_gumbel_index[i][0] = self.gumbel_sample(gumbel_logits[i])
                            # att_cf_decoder[i] = (F.gumbel_softmax(gumbel_logits[i], tau=0.1, hard=False)) * g_decoder[j][-1]
                            if self.random_bu:
                                att_cf_decoder[i] = self.Add_random_noise(g_decoder[j][-1], self.noise_factor)
                            else:
                                att_cf_decoder[i] = self.Add_noise(g_decoder[j][-1], sample_gumbel_index[i][0], self.noise_factor)

        # define task prediction layers
        t1_pred = F.softmax(self.fc_1(self.pred_task_1(atten_decoder[0][-1][-1]).view(self.batch_size, -1)), dim=1)
        t2_pred = F.softmax(self.fc_2(self.pred_task_2(atten_decoder[1][-1][-1]).view(self.batch_size, -1)), dim=1)

        if self.training:
            # task1 and task2
            gumbel_sample_index_1_2  = (sample_gumbel_index[0][0] + sample_gumbel_index[1][0]) / 2
            if self.random_bu:
                att_cf_decoder_1_2 = self.Add_random_noise(g_decoder[-1][-1], self.noise_factor)
            else:
                att_cf_decoder_1_2 = self.Add_noise(g_decoder[-1][-1], gumbel_sample_index_1_2,
                                                    self.noise_factor)

            del gumbel_logits, gumbel_sample_index_1_2

            t1_cf_pred = F.softmax(self.fc_1(self.pred_task_1(att_cf_decoder[0]).view(self.batch_size, -1)), dim=1)
            t2_cf_pred = F.softmax(self.fc_2(self.pred_task_2(att_cf_decoder[1]).view(self.batch_size, -1)), dim=1)

            t_1_2_to_1_cf_pred = F.softmax(self.fc_1(self.pred_task_1(att_cf_decoder_1_2).view(self.batch_size, -1)),
                                           dim=1)
            t_1_2_to_2_cf_pred = F.softmax(self.fc_2(self.pred_task_2(att_cf_decoder_1_2).view(self.batch_size, -1)),
                                           dim=1)

            del att_cf_decoder_1_2
            return [t1_pred, t2_pred], [t1_cf_pred, t2_cf_pred], [t_1_2_to_1_cf_pred, t_1_2_to_2_cf_pred]


        else:
            return [t1_pred, t2_pred]


class New_SegNet_Gumbel_Softmax(nn.Module):
    def __init__(self, k, patch_size, Tau_coe, device):
        super(New_SegNet_Gumbel_Softmax, self).__init__()
        # initialise network parameters
        filter = [8, 16, 32, 64, 64]
        self.class_nb = 10
        self.batch_size = 256
        self.k = k
        self.patch_size = patch_size
        self.Tau_coe = Tau_coe
        self.device = device
        self.noise_factor = 0.3

        # define encoder decoder layers
        self.encoder_block = nn.ModuleList([self.conv_layer([1, filter[0]])])
        self.decoder_block = nn.ModuleList([self.conv_layer([filter[0], filter[0]])])
        for i in range(4):
            self.encoder_block.append(self.conv_layer([filter[i], filter[i + 1]]))
            self.decoder_block.append(self.conv_layer([filter[i + 1], filter[i]]))

        # define convolution layer
        self.conv_block_enc = nn.ModuleList([self.conv_layer([filter[0], filter[0]])])
        self.conv_block_dec = nn.ModuleList([self.conv_layer([filter[0], filter[0]])])
        for i in range(4):
            if i == 0:
                self.conv_block_enc.append(self.conv_layer([filter[i + 1], filter[i + 1]]))
                self.conv_block_dec.append(self.conv_layer([filter[i], filter[i]]))
            else:
                self.conv_block_enc.append(nn.Sequential(self.conv_layer([filter[i + 1], filter[i + 1]]),
                                                         self.conv_layer([filter[i + 1], filter[i + 1]])))
                self.conv_block_dec.append(nn.Sequential(self.conv_layer([filter[i], filter[i]]),
                                                         self.conv_layer([filter[i], filter[i]])))

        # define task attention layers
        self.encoder_att = nn.ModuleList([nn.ModuleList([self.att_layer([filter[0], filter[0], filter[0]])])])
        self.decoder_att = nn.ModuleList([nn.ModuleList([self.att_layer([2 * filter[0], filter[0], filter[0]])])])
        self.encoder_block_att = nn.ModuleList([self.conv_layer([filter[0], filter[1]])])
        self.decoder_block_att = nn.ModuleList([self.conv_layer([filter[0], filter[0]])])

        for j in range(2):
            if j < 1:
                self.encoder_att.append(nn.ModuleList([self.att_layer([filter[0], filter[0], filter[0]])]))
                self.decoder_att.append(nn.ModuleList([self.att_layer([2 * filter[0], filter[0], filter[0]])]))
            for i in range(4):
                self.encoder_att[j].append(self.att_layer([2 * filter[i + 1], filter[i + 1], filter[i + 1]]))
                self.decoder_att[j].append(self.att_layer([filter[i + 1] + filter[i], filter[i], filter[i]]))

        for i in range(4):
            if i < 3:
                self.encoder_block_att.append(self.conv_layer([filter[i + 1], filter[i + 2]]))
                self.decoder_block_att.append(self.conv_layer([filter[i + 1], filter[i]]))
            else:
                self.encoder_block_att.append(self.conv_layer([filter[i + 1], filter[i + 1]]))
                self.decoder_block_att.append(self.conv_layer([filter[i + 1], filter[i + 1]]))

        self.pred_task_1 = self.conv_layer([filter[0], 1], pred=True)
        self.pred_task_2 = self.conv_layer([filter[0], 1], pred=True)
        self.pred_task_logits = self.out_logits_conv_layer([filter[0], 1])

        # define pooling and unpooling functions
        self.down_sampling = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.up_sampling = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.fc_1 = nn.Linear(64 * 64, self.class_nb)
        self.fc_2 = nn.Linear(64 * 64, self.class_nb)
        self.logsigma = nn.Parameter(torch.FloatTensor([-0.5, -0.5]))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def shared_modules(self):
        return [self.encoder_block, self.decoder_block,
                self.conv_block_enc, self.conv_block_dec,
                self.encoder_block_att, self.decoder_block_att,
                self.down_sampling, self.up_sampling]

    def zero_grad_shared_modules(self):
        for mm in self.shared_modules():
            mm.zero_grad()

    def conv_layer(self, channel, pred=False):
        if not pred:
            conv_block = nn.Sequential(
                nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=3, padding=1),
                nn.BatchNorm2d(num_features=channel[1]),
                nn.ReLU(inplace=True),
            )
        else:
            conv_block = nn.Sequential(
                nn.Conv2d(in_channels=channel[0], out_channels=channel[0], kernel_size=3, padding=1),
                nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=1, padding=0),
            )
        return conv_block

    def out_logits_conv_layer(self, channel):
        conv_block = nn.Sequential(
            nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Conv2d(in_channels=channel[1], out_channels=1, kernel_size=1),
            # nn.MaxPool2d(kernel_size=2, stride=2),
        )
        return conv_block

    def att_layer(self, channel):
        att_block = nn.Sequential(
            nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=1, padding=0),
            nn.BatchNorm2d(channel[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=channel[1], out_channels=channel[2], kernel_size=1, padding=0),
            nn.BatchNorm2d(channel[2]),
            nn.Sigmoid(),
        )
        return att_block

    def sample(self, logits):
        # print(logits.size())
        batch_size, channels, d1, d2 = logits.size()
        if self.batch_size != batch_size:
            self.batch_size = batch_size
        dims_s = self.batch_size * channels * d1 * d2
        # upsample_op = nn.Upsample(scale_factor=self.patch_size, mode='nearest')
        K = self.batch_size * self.k * channels
        sample_logits = torch.zeros_like(logits.view(-1, dims_s))  # (-1, bactch * logits0 * logits1)

        logits = torch.reshape(logits, (-1, dims_s))  # (-1, bactch * logits0 * logits1)
        vals, indics = torch.topk(logits, K, dim=-1)
        sample_logits[0, indics[0, :]] = 1
        del vals, indics

        sample_logits = torch.reshape(sample_logits, (self.batch_size, channels, d1, d2))
        # imp = upsample_op(sample_logits)
        del logits
        return sample_logits

    def Sample_gumbel(self, logits):
        batch_size, channels, d1, d2 = logits.size()
        #d1, d2 = logits.shape[2], logits.shape[3]
        dims = channels * d1 * d2
        #print("channels = ", channels)
        if self.training == True:
            softmax = nn.Softmax(dim=-1)
            logits = logits.view(self.batch_size, -1).unsqueeze(1)
            #print("logits.size()", logits.size())
            unif_shape = [self.batch_size, self.k, dims]
            #print("unif_shape", unif_shape)
            uniform = (1 - 0) * torch.rand(unif_shape)

            gumbel = - torch.log(-torch.log(uniform)).to(self.device)
            #print("gumbel.size()", gumbel.size())
            noisy_logits = (gumbel + logits) / self.Tau_coe
            samples = softmax(noisy_logits)
            samples, samples_ids = torch.max(samples, dim=1)
            #print("samples.size()", samples.size())

            del logits, unif_shape, uniform, gumbel, noisy_logits, samples_ids
            samples = torch.reshape(samples, (self.batch_size, channels, d1, d2))
            #selected_subset = torch.unsqueeze(samples, dim=1)
            imp = samples

            del samples
        else:
            imp = self.sample(logits)
        return imp

    def gumbel_sample(self, logits):
        batch_size, channels, d1, d2 = logits.size()
        dims = d1 * d2
        #imp_index = self.sample(gumbel_samples)
        #noise_index = torch.ones_like(imp_index) - imp_index

        gumbel_samples = F.gumbel_softmax(logits, tau=self.Tau_coe, hard=False)

        return gumbel_samples

    def Add_noise(self, feas, imp_scores, noise_scores, noise_factor):
        #noisy_cf = feas + torch.randn_like(feas) * noise_scores * noise_factor
        #noisy_cf = feas * imp_scores + torch.randn_like(feas) * noise_scores
        #print('feas.size()=',feas.size())
        #print('imp_scores()=', imp_scores.size())
        noisy_cf = feas * imp_scores + torch.randn_like(feas * imp_scores) * noise_factor
        return noisy_cf

    def Add_random_noise(self, feas, noise_factor):
        noisy_cf = feas + torch.randn_like(feas) * noise_factor
        return noisy_cf

    def Union_task_gumbel_index(self, task1_index, task2_index):

        batch_size, channels, d1, d2 = task1_index.size()
        dims = batch_size * channels * d1 * d2
        gumbel_index_Union_1_2 = (task1_index + task2_index).view(dims)
        # print("gumbel_index_Union_1_2.size()", gumbel_index_Union_1_2.size())
        gumbel_sample_index_1_2 = torch.zeros_like(gumbel_index_Union_1_2)
        # print("gumbel_sample_index_1_2.size()", gumbel_sample_index_1_2.size())

        nonzero_index = torch.nonzero(gumbel_index_Union_1_2)
        # print("nonzero_index.size()", nonzero_index.size())
        # print("nonzero_index", nonzero_index)
        gumbel_sample_index_1_2[nonzero_index] = 1
        noise_index_1_2 = torch.ones_like(gumbel_sample_index_1_2) - gumbel_sample_index_1_2

        gumbel_sample_index_1_2 = gumbel_sample_index_1_2.view(batch_size, channels, d1, d2)
        noise_index_1_2 = noise_index_1_2.view(batch_size, channels, d1, d2)

        del batch_size, channels, d1, d2, gumbel_index_Union_1_2, nonzero_index

        return gumbel_sample_index_1_2, noise_index_1_2

    def forward(self, x, training=True, random_bu=False):
        self.batch_size = x.size(0)
        self.training = training
        self.random_bu = random_bu
        g_encoder, g_decoder, g_maxpool, g_upsampl, indices = ([0] * 5 for _ in range(5))
        for i in range(5):
            g_encoder[i], g_decoder[-i - 1] = ([0] * 2 for _ in range(2))

        # define attention list for tasks
        atten_encoder, atten_decoder = ([0] * 2 for _ in range(2))
        att_cf_decoder, gumbel_logits = ([0] * 2 for _ in range(2))
        sample_gumbel_index = [0] * 2
        for i in range(2):
            atten_encoder[i], atten_decoder[i] = ([0] * 5 for _ in range(2))
            sample_gumbel_index[i] = [0] * 2
        for i in range(2):
            for j in range(5):
                atten_encoder[i][j], atten_decoder[i][j] = ([0] * 3 for _ in range(2))

        # define global shared network
        for i in range(5):
            if i == 0:
                g_encoder[i][0] = self.encoder_block[i](x)
                g_encoder[i][1] = self.conv_block_enc[i](g_encoder[i][0])
                g_maxpool[i], indices[i] = self.down_sampling(g_encoder[i][1])
            else:
                g_encoder[i][0] = self.encoder_block[i](g_maxpool[i - 1])
                g_encoder[i][1] = self.conv_block_enc[i](g_encoder[i][0])
                g_maxpool[i], indices[i] = self.down_sampling(g_encoder[i][1])

        for i in range(5):
            if i == 0:
                g_upsampl[i] = self.up_sampling(g_maxpool[-1], indices[-i - 1])
                g_decoder[i][0] = self.decoder_block[-i - 1](g_upsampl[i])
                g_decoder[i][1] = self.conv_block_dec[-i - 1](g_decoder[i][0])
            else:
                g_upsampl[i] = self.up_sampling(g_decoder[i - 1][-1], indices[-i - 1])
                g_decoder[i][0] = self.decoder_block[-i - 1](g_upsampl[i])
                g_decoder[i][1] = self.conv_block_dec[-i - 1](g_decoder[i][0])

        # define task dependent attention module
        for i in range(2):
            for j in range(5):
                if j == 0:
                    atten_encoder[i][j][0] = self.encoder_att[i][j](g_encoder[j][0])
                    atten_encoder[i][j][1] = (atten_encoder[i][j][0]) * g_encoder[j][1]
                    atten_encoder[i][j][2] = self.encoder_block_att[j](atten_encoder[i][j][1])
                    atten_encoder[i][j][2] = F.max_pool2d(atten_encoder[i][j][2], kernel_size=2, stride=2)
                else:
                    atten_encoder[i][j][0] = self.encoder_att[i][j](
                        torch.cat((g_encoder[j][0], atten_encoder[i][j - 1][2]), dim=1))
                    atten_encoder[i][j][1] = (atten_encoder[i][j][0]) * g_encoder[j][1]
                    atten_encoder[i][j][2] = self.encoder_block_att[j](atten_encoder[i][j][1])
                    atten_encoder[i][j][2] = F.max_pool2d(atten_encoder[i][j][2], kernel_size=2, stride=2)

            for j in range(5):
                if j == 0:
                    atten_decoder[i][j][0] = F.interpolate(atten_encoder[i][-1][-1], scale_factor=2, mode='bilinear',
                                                           align_corners=True)
                    atten_decoder[i][j][0] = self.decoder_block_att[-j - 1](atten_decoder[i][j][0])
                    atten_decoder[i][j][1] = self.decoder_att[i][-j - 1](
                        torch.cat((g_upsampl[j], atten_decoder[i][j][0]), dim=1))
                    atten_decoder[i][j][2] = (atten_decoder[i][j][1]) * g_decoder[j][-1]
                else:
                    atten_decoder[i][j][0] = F.interpolate(atten_decoder[i][j - 1][2], scale_factor=2, mode='bilinear',
                                                           align_corners=True)
                    atten_decoder[i][j][0] = self.decoder_block_att[-j - 1](atten_decoder[i][j][0])
                    atten_decoder[i][j][1] = self.decoder_att[i][-j - 1](
                        torch.cat((g_upsampl[j], atten_decoder[i][j][0]), dim=1))
                    atten_decoder[i][j][2] = (atten_decoder[i][j][1]) * g_decoder[j][-1]
                    if self.training:
                        if j == 4:
                            gumbel_logits[i] = atten_decoder[i][j][1]
                            sample_gumbel_index[i][0] = self.gumbel_sample(gumbel_logits[i])
                            #sample_gumbel_index[i][1] = self.Sample_gumbel(gumbel_logits[i])
                            # att_cf_decoder[i] = (F.gumbel_softmax(gumbel_logits[i], tau=0.1, hard=False)) * g_decoder[j][-1]
                            if self.random_bu:
                                att_cf_decoder[i] = self.Add_random_noise(g_decoder[j][-1], self.noise_factor)
                            else:
                                att_cf_decoder[i] = self.Add_noise(g_decoder[j][-1], sample_gumbel_index[i][0],
                                                                   sample_gumbel_index[i][0], self.noise_factor)

        # define task prediction layers
        t1_pred = F.softmax(self.fc_1(self.pred_task_1(atten_decoder[0][-1][-1]).view(self.batch_size, -1)), dim=1)
        t2_pred = F.softmax(self.fc_2(self.pred_task_2(atten_decoder[1][-1][-1]).view(self.batch_size, -1)), dim=1)

        if self.training:
            # task1 and task2
            #gumbel_sample_index_1_2, noise_index_1_2 = self.Union_task_gumbel_index(sample_gumbel_index[0][0], sample_gumbel_index[1][0])
            gumbel_sample_index_1_2  = (sample_gumbel_index[0][0] + sample_gumbel_index[1][0]) / 2
            if self.random_bu:
                att_cf_decoder_1_2 = self.Add_random_noise(g_decoder[-1][-1], self.noise_factor)
            else:
                att_cf_decoder_1_2 = self.Add_noise(g_decoder[-1][-1], gumbel_sample_index_1_2, gumbel_sample_index_1_2,
                                                    self.noise_factor)

            del gumbel_logits, gumbel_sample_index_1_2

            t1_cf_pred = F.softmax(self.fc_1(self.pred_task_1(att_cf_decoder[0]).view(self.batch_size, -1)), dim=1)
            t2_cf_pred = F.softmax(self.fc_2(self.pred_task_2(att_cf_decoder[1]).view(self.batch_size, -1)), dim=1)

            t_1_2_to_1_cf_pred = F.softmax(self.fc_1(self.pred_task_1(att_cf_decoder_1_2).view(self.batch_size, -1)),
                                           dim=1)
            t_1_2_to_2_cf_pred = F.softmax(self.fc_2(self.pred_task_2(att_cf_decoder_1_2).view(self.batch_size, -1)),
                                           dim=1)

            del att_cf_decoder_1_2
            return [t1_pred, t2_pred], [t1_cf_pred, t2_cf_pred], [t_1_2_to_1_cf_pred, t_1_2_to_2_cf_pred]


        else:
            return [t1_pred, t2_pred]




class SegNet_K_Gumbel_Softmax(nn.Module):
    def __init__(self, k, patch_size, Tau_coe, device):
        super(SegNet_K_Gumbel_Softmax, self).__init__()
        # initialise network parameters
        filter = [8, 16, 32, 64, 64]
        self.class_nb = 10
        self.batch_size = 256
        self.k = k
        self.patch_size = patch_size
        self.Tau_coe = Tau_coe
        self.device = device
        self.noise_factor = 0.3

        # define encoder decoder layers
        self.encoder_block = nn.ModuleList([self.conv_layer([1, filter[0]])])
        self.decoder_block = nn.ModuleList([self.conv_layer([filter[0], filter[0]])])
        for i in range(4):
            self.encoder_block.append(self.conv_layer([filter[i], filter[i + 1]]))
            self.decoder_block.append(self.conv_layer([filter[i + 1], filter[i]]))

        # define convolution layer
        self.conv_block_enc = nn.ModuleList([self.conv_layer([filter[0], filter[0]])])
        self.conv_block_dec = nn.ModuleList([self.conv_layer([filter[0], filter[0]])])
        for i in range(4):
            if i == 0:
                self.conv_block_enc.append(self.conv_layer([filter[i + 1], filter[i + 1]]))
                self.conv_block_dec.append(self.conv_layer([filter[i], filter[i]]))
            else:
                self.conv_block_enc.append(nn.Sequential(self.conv_layer([filter[i + 1], filter[i + 1]]),
                                                         self.conv_layer([filter[i + 1], filter[i + 1]])))
                self.conv_block_dec.append(nn.Sequential(self.conv_layer([filter[i], filter[i]]),
                                                         self.conv_layer([filter[i], filter[i]])))

        # define task attention layers
        self.encoder_att = nn.ModuleList([nn.ModuleList([self.att_layer([filter[0], filter[0], filter[0]])])])
        self.decoder_att = nn.ModuleList([nn.ModuleList([self.att_layer([2 * filter[0], filter[0], filter[0]])])])
        self.encoder_block_att = nn.ModuleList([self.conv_layer([filter[0], filter[1]])])
        self.decoder_block_att = nn.ModuleList([self.conv_layer([filter[0], filter[0]])])

        for j in range(2):
            if j < 1:
                self.encoder_att.append(nn.ModuleList([self.att_layer([filter[0], filter[0], filter[0]])]))
                self.decoder_att.append(nn.ModuleList([self.att_layer([2 * filter[0], filter[0], filter[0]])]))
            for i in range(4):
                self.encoder_att[j].append(self.att_layer([2 * filter[i + 1], filter[i + 1], filter[i + 1]]))
                self.decoder_att[j].append(self.att_layer([filter[i + 1] + filter[i], filter[i], filter[i]]))

        for i in range(4):
            if i < 3:
                self.encoder_block_att.append(self.conv_layer([filter[i + 1], filter[i + 2]]))
                self.decoder_block_att.append(self.conv_layer([filter[i + 1], filter[i]]))
            else:
                self.encoder_block_att.append(self.conv_layer([filter[i + 1], filter[i + 1]]))
                self.decoder_block_att.append(self.conv_layer([filter[i + 1], filter[i + 1]]))

        self.pred_task_1 = self.conv_layer([filter[0], 1], pred=True)
        self.pred_task_2 = self.conv_layer([filter[0], 1], pred=True)
        self.pred_task_logits = self.out_logits_conv_layer([filter[0], 1])
        # self.pred_task3 = self.conv_layer([filter[0], 3], pred=True)out_logits_conv_layer

        # define pooling and unpooling functions
        self.down_sampling = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.up_sampling = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.fc_1 = nn.Linear(64 * 64, self.class_nb)
        self.fc_2 = nn.Linear(64 * 64, self.class_nb)
        self.logsigma = nn.Parameter(torch.FloatTensor([-0.5, -0.5]))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def shared_modules(self):
        return [self.encoder_block, self.decoder_block,
                self.conv_block_enc, self.conv_block_dec,
                self.encoder_block_att, self.decoder_block_att,
                self.down_sampling, self.up_sampling]

    def zero_grad_shared_modules(self):
        for mm in self.shared_modules():
            mm.zero_grad()

    def conv_layer(self, channel, pred=False):
        if not pred:
            conv_block = nn.Sequential(
                nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=3, padding=1),
                nn.BatchNorm2d(num_features=channel[1]),
                nn.ReLU(inplace=True),
            )
        else:
            conv_block = nn.Sequential(
                nn.Conv2d(in_channels=channel[0], out_channels=channel[0], kernel_size=3, padding=1),
                nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=1, padding=0),
            )
        return conv_block

    def out_logits_conv_layer(self, channel):
        conv_block = nn.Sequential(
            nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Conv2d(in_channels=channel[1], out_channels=1, kernel_size=1),
            # nn.MaxPool2d(kernel_size=2, stride=2),
        )
        return conv_block

    def att_layer(self, channel):
        att_block = nn.Sequential(
            nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=1, padding=0),
            nn.BatchNorm2d(channel[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=channel[1], out_channels=channel[2], kernel_size=1, padding=0),
            nn.BatchNorm2d(channel[2]),
            nn.Sigmoid(),
        )
        return att_block

    def sample(self, logits):
        # print(logits.size())
        batch_size, channels, d1, d2 = logits.size()
        if self.batch_size != batch_size:
            self.batch_size = batch_size
        dims_s = self.batch_size * channels * d1 * d2
        # upsample_op = nn.Upsample(scale_factor=self.patch_size, mode='nearest')
        K = self.batch_size * self.k * channels
        sample_logits = torch.zeros_like(logits.view(-1, dims_s))  # (-1, bactch * logits0 * logits1)

        logits = torch.reshape(logits, (-1, dims_s))  # (-1, bactch * logits0 * logits1)
        vals, indics = torch.topk(logits, K, dim=-1)
        sample_logits[0, indics[0, :]] = 1
        del vals, indics

        sample_logits = torch.reshape(sample_logits, (self.batch_size, channels, d1, d2))
        # imp = upsample_op(sample_logits)
        del logits
        return sample_logits

    def Sample_gumbel(self, logits):
        batch_size, channels, d1, d2 = logits.size()
        #d1, d2 = logits.shape[2], logits.shape[3]
        dims = channels * d1 * d2
        #print("channels = ", channels)
        if self.training == True:
            softmax = nn.Softmax(dim=-1)
            logits = logits.view(self.batch_size, -1).unsqueeze(1)
            #print("logits.size()", logits.size())
            unif_shape = [self.batch_size, self.k, dims]
            #print("unif_shape", unif_shape)
            uniform = (1 - 0) * torch.rand(unif_shape)

            gumbel = - torch.log(-torch.log(uniform)).to(self.device)
            #print("gumbel.size()", gumbel.size())
            noisy_logits = (gumbel + logits) / self.Tau_coe
            samples = softmax(noisy_logits)
            samples, samples_ids = torch.max(samples, dim=1)
            #print("samples.size()", samples.size())

            del logits, unif_shape, uniform, gumbel, noisy_logits, samples_ids
            samples = torch.reshape(samples, (self.batch_size, channels, d1, d2))
            #selected_subset = torch.unsqueeze(samples, dim=1)
            imp = samples

            del samples
        else:
            imp = self.sample(logits)
        return imp

    def gumbel_sample(self, logits):
        batch_size, channels, d1, d2 = logits.size()
        dims = d1 * d2
        #print('logits size', logits.size())
        
        #print('one time GumbelSoftmax size', F.gumbel_softmax(logits, tau=self.Tau_coe, hard=False).size())
        gumbel_sample_k_time = [F.gumbel_softmax(logits, tau=self.Tau_coe, hard=False) for _ in range(self.k)]
        #gumbel_samples = F.gumbel_softmax(logits, tau=self.Tau_coe, hard=False)
        gumbel_sample_k_times = torch.stack(gumbel_sample_k_time,dim=0)
        #print('gumbel_sample_k_times size', gumbel_sample_k_times.size())
        gumbel_samples, ids = torch.max(gumbel_sample_k_times, dim=0)
        #print('gumbel_samples size', gumbel_samples.size())
        #gumbel_samples = torch.squeeze(gumbel_samples, dim=0)
        #print('gumbel_samples size', gumbel_samples.size())
        #imp_index = self.sample(gumbel_samples)
        #noise_index = torch.ones_like(imp_index) - imp_index

        #gumbel_samples = F.gumbel_softmax(logits, tau=self.Tau_coe, hard=False)
        del ids

        return gumbel_samples

    def Add_noise(self, feas, imp_scores, noise_scores, noise_factor):
        #noisy_cf = feas + torch.randn_like(feas) * noise_scores * noise_factor
        #noisy_cf = feas * imp_scores + torch.randn_like(feas) * noise_scores
        #print('feas.size()=',feas.size())
        #print('imp_scores()=', imp_scores.size())
        noisy_cf = feas * imp_scores + torch.randn_like(feas * imp_scores) * noise_factor
        return noisy_cf

    def Add_random_noise(self, feas, noise_factor):
        noisy_cf = feas + torch.randn_like(feas) * noise_factor
        return noisy_cf

    def Union_task_gumbel_index(self, task1_index, task2_index):

        batch_size, channels, d1, d2 = task1_index.size()
        dims = batch_size * channels * d1 * d2
        gumbel_index_Union_1_2 = (task1_index + task2_index).view(dims)
        # print("gumbel_index_Union_1_2.size()", gumbel_index_Union_1_2.size())
        gumbel_sample_index_1_2 = torch.zeros_like(gumbel_index_Union_1_2)
        # print("gumbel_sample_index_1_2.size()", gumbel_sample_index_1_2.size())

        nonzero_index = torch.nonzero(gumbel_index_Union_1_2)
        # print("nonzero_index.size()", nonzero_index.size())
        # print("nonzero_index", nonzero_index)
        gumbel_sample_index_1_2[nonzero_index] = 1
        noise_index_1_2 = torch.ones_like(gumbel_sample_index_1_2) - gumbel_sample_index_1_2

        gumbel_sample_index_1_2 = gumbel_sample_index_1_2.view(batch_size, channels, d1, d2)
        noise_index_1_2 = noise_index_1_2.view(batch_size, channels, d1, d2)

        del batch_size, channels, d1, d2, gumbel_index_Union_1_2, nonzero_index

        return gumbel_sample_index_1_2, noise_index_1_2

    def forward(self, x, training=True, random_bu=False):
        self.batch_size = x.size(0)
        self.training = training
        self.random_bu = random_bu
        g_encoder, g_decoder, g_maxpool, g_upsampl, indices = ([0] * 5 for _ in range(5))
        for i in range(5):
            g_encoder[i], g_decoder[-i - 1] = ([0] * 2 for _ in range(2))

        # define attention list for tasks
        atten_encoder, atten_decoder = ([0] * 2 for _ in range(2))
        att_cf_decoder, gumbel_logits = ([0] * 2 for _ in range(2))
        sample_gumbel_index = [0] * 2
        for i in range(2):
            atten_encoder[i], atten_decoder[i] = ([0] * 5 for _ in range(2))
            sample_gumbel_index[i] = [0] * 2
        for i in range(2):
            for j in range(5):
                atten_encoder[i][j], atten_decoder[i][j] = ([0] * 3 for _ in range(2))

        # define global shared network
        for i in range(5):
            if i == 0:
                g_encoder[i][0] = self.encoder_block[i](x)
                g_encoder[i][1] = self.conv_block_enc[i](g_encoder[i][0])
                g_maxpool[i], indices[i] = self.down_sampling(g_encoder[i][1])
            else:
                g_encoder[i][0] = self.encoder_block[i](g_maxpool[i - 1])
                g_encoder[i][1] = self.conv_block_enc[i](g_encoder[i][0])
                g_maxpool[i], indices[i] = self.down_sampling(g_encoder[i][1])

        for i in range(5):
            if i == 0:
                g_upsampl[i] = self.up_sampling(g_maxpool[-1], indices[-i - 1])
                g_decoder[i][0] = self.decoder_block[-i - 1](g_upsampl[i])
                g_decoder[i][1] = self.conv_block_dec[-i - 1](g_decoder[i][0])
            else:
                g_upsampl[i] = self.up_sampling(g_decoder[i - 1][-1], indices[-i - 1])
                g_decoder[i][0] = self.decoder_block[-i - 1](g_upsampl[i])
                g_decoder[i][1] = self.conv_block_dec[-i - 1](g_decoder[i][0])

        # define task dependent attention module
        for i in range(2):
            for j in range(5):
                if j == 0:
                    atten_encoder[i][j][0] = self.encoder_att[i][j](g_encoder[j][0])
                    atten_encoder[i][j][1] = (atten_encoder[i][j][0]) * g_encoder[j][1]
                    atten_encoder[i][j][2] = self.encoder_block_att[j](atten_encoder[i][j][1])
                    atten_encoder[i][j][2] = F.max_pool2d(atten_encoder[i][j][2], kernel_size=2, stride=2)
                else:
                    atten_encoder[i][j][0] = self.encoder_att[i][j](
                        torch.cat((g_encoder[j][0], atten_encoder[i][j - 1][2]), dim=1))
                    atten_encoder[i][j][1] = (atten_encoder[i][j][0]) * g_encoder[j][1]
                    atten_encoder[i][j][2] = self.encoder_block_att[j](atten_encoder[i][j][1])
                    atten_encoder[i][j][2] = F.max_pool2d(atten_encoder[i][j][2], kernel_size=2, stride=2)

            for j in range(5):
                if j == 0:
                    atten_decoder[i][j][0] = F.interpolate(atten_encoder[i][-1][-1], scale_factor=2, mode='bilinear',
                                                           align_corners=True)
                    atten_decoder[i][j][0] = self.decoder_block_att[-j - 1](atten_decoder[i][j][0])
                    atten_decoder[i][j][1] = self.decoder_att[i][-j - 1](
                        torch.cat((g_upsampl[j], atten_decoder[i][j][0]), dim=1))
                    atten_decoder[i][j][2] = (atten_decoder[i][j][1]) * g_decoder[j][-1]
                else:
                    atten_decoder[i][j][0] = F.interpolate(atten_decoder[i][j - 1][2], scale_factor=2, mode='bilinear',
                                                           align_corners=True)
                    atten_decoder[i][j][0] = self.decoder_block_att[-j - 1](atten_decoder[i][j][0])
                    atten_decoder[i][j][1] = self.decoder_att[i][-j - 1](
                        torch.cat((g_upsampl[j], atten_decoder[i][j][0]), dim=1))
                    atten_decoder[i][j][2] = (atten_decoder[i][j][1]) * g_decoder[j][-1]
                    if self.training:
                        if j == 4:
                            gumbel_logits[i] = atten_decoder[i][j][1]
                            sample_gumbel_index[i][0] = self.gumbel_sample(gumbel_logits[i])
                            #sample_gumbel_index[i][1] = self.Sample_gumbel(gumbel_logits[i])
                            # att_cf_decoder[i] = (F.gumbel_softmax(gumbel_logits[i], tau=0.1, hard=False)) * g_decoder[j][-1]
                            if self.random_bu:
                                att_cf_decoder[i] = self.Add_random_noise(g_decoder[j][-1], self.noise_factor)
                            else:
                                att_cf_decoder[i] = self.Add_noise(g_decoder[j][-1], sample_gumbel_index[i][0],
                                                                   sample_gumbel_index[i][0], self.noise_factor)

        # define task prediction layers
        t1_pred = F.softmax(self.fc_1(self.pred_task_1(atten_decoder[0][-1][-1]).view(self.batch_size, -1)), dim=1)
        t2_pred = F.softmax(self.fc_2(self.pred_task_2(atten_decoder[1][-1][-1]).view(self.batch_size, -1)), dim=1)

        if self.training:
            # task1 and task2
            #gumbel_sample_index_1_2, noise_index_1_2 = self.Union_task_gumbel_index(sample_gumbel_index[0][0], sample_gumbel_index[1][0])
            gumbel_sample_index_1_2  = (sample_gumbel_index[0][0] + sample_gumbel_index[1][0]) / 2
            if self.random_bu:
                att_cf_decoder_1_2 = self.Add_random_noise(g_decoder[-1][-1], self.noise_factor)
            else:
                att_cf_decoder_1_2 = self.Add_noise(g_decoder[-1][-1], gumbel_sample_index_1_2, gumbel_sample_index_1_2,
                                                    self.noise_factor)

            del gumbel_logits, gumbel_sample_index_1_2

            t1_cf_pred = F.softmax(self.fc_1(self.pred_task_1(att_cf_decoder[0]).view(self.batch_size, -1)), dim=1)
            t2_cf_pred = F.softmax(self.fc_2(self.pred_task_2(att_cf_decoder[1]).view(self.batch_size, -1)), dim=1)

            t_1_2_to_1_cf_pred = F.softmax(self.fc_1(self.pred_task_1(att_cf_decoder_1_2).view(self.batch_size, -1)),
                                           dim=1)
            t_1_2_to_2_cf_pred = F.softmax(self.fc_2(self.pred_task_2(att_cf_decoder_1_2).view(self.batch_size, -1)),
                                           dim=1)

            del att_cf_decoder_1_2
            return [t1_pred, t2_pred], [t1_cf_pred, t2_cf_pred], [t_1_2_to_1_cf_pred, t_1_2_to_2_cf_pred]


        else:
            return [t1_pred, t2_pred]


