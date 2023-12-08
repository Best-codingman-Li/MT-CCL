# coding:utf-8
import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, Sampler
from torchvision import datasets, transforms



#dataset

train_txt_path = os.path.join("/code/CFS_DoubleMnist/double_mnist", "train.txt")
train_dir = os.path.join("/code/CFS_DoubleMnist/double_mnist", "train")

test_txt_path = os.path.join("/code/CFS_DoubleMnist/double_mnist", "test.txt")
test_dir = os.path.join("/code/CFS_DoubleMnist/double_mnist", "test")

def gen_txt(txt_path, img_dir):
    f = open(txt_path, 'w')

    for root, s_dirs, _ in os.walk(img_dir, topdown=True):  
        for sub_dir in s_dirs:
            i_dir = os.path.join(root, sub_dir)  
            img_list = os.listdir(i_dir)  
            for i in range(len(img_list)):
                if not img_list[i].endswith('png'):  
                    continue
                label = img_list[i].split('_')[1].strip('.png')
                img_path = os.path.join(i_dir, img_list[i])
                line = img_path + ' ' + str(label[0]) + ' ' + str(label[1]) + '\n'
                f.write(line)
    f.close()



class MyDataset(Dataset):
    def __init__(self, txt_path, transform=None, target_transform=None):
        fh = open(txt_path, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], int(words[1]), int(words[2])))

        self.imgs = imgs  
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        fn, label1, label2 = self.imgs[index]
        img = Image.open(fn)

        if self.transform is not None:
            img = self.transform(img)  
            
        tar = np.array([label1, label2])
        return img, torch.from_numpy(tar)

    def __len__(self):
        return len(self.imgs)

    def print_imges(self):
        print(len(self.imgs))
        for i in range(len(self.imgs)):

            print(self.imgs[i])




