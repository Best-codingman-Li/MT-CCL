
import pdb
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torchvision.transforms as transforms
import torch.utils.data as Data
from PIL import Image
import numpy as np
import os
import torch.backends.cudnn as cudnn
img_root = '/code/img_align_celeba'
image_text = '/code/Anno/list_attr_celeba.txt'
train_text = '/code/Anno/train_attr_celeba.txt'
test_text = '/code/Anno/test_attr_celeba.txt'
batch_size = 128
 
save_folder = './celeba/'
if not os.path.exists(save_folder):
    os.mkdir(save_folder)
 
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')
 
 
def default_loader(path):
    try:
        img = Image.open(path)
        return img.convert('RGB')
    except:
        print("Can not open {0}".format(path))
 
def train_test_split(src_text, train_text, test_text):
    
    fs = open(src_text, 'r')
    ftr = open(train_text, 'w')
    fte = open(test_text, 'w')
    i = 0
    for line in fs.readlines():
        if len(line.split()) != 41:
            continue
        i += 1
        if i <= 141819:
            ftr.write(line)
        elif i <= 202599 :
            fte.write(line)
        else:
            continue
            
    fs.close()
    ftr.close()
    fte.close()
            
class myDataset(Data.DataLoader):
    def __init__(self, img_dir, img_txt, transform=None, loader=default_loader):
        img_list = []
        img_labels = []
        fp = open(img_txt, 'r')
        for line in fp.readlines():
            if len(line.split()) != 41:
                continue
            img_list.append(line.split()[0])
            img_label_single = []
            for value in line.split()[1:]:
                if value == '-1':
                    img_label_single.append(0)
                if value == '1':
                    img_label_single.append(1)
            img_labels.append(img_label_single)
        self.imgs = [os.path.join(img_dir, file) for file in img_list]
        self.labels = img_labels
        self.transform = transform
        self.loader = loader
    def __len__(self):
        return len(self.imgs)
    def __getitem__(self, index):
        img_path = self.imgs[index]
        label = torch.from_numpy(np.array(self.labels[index], dtype=np.int64))
        img = self.loader(img_path)
        if self.transform is not None:
            try:
                img = self.transform(img)
            except:
                print('Cannot transform image: {}'.format(img_path))
        return img, label
 
transform = transforms.Compose([
    transforms.Resize(40),
    transforms.CenterCrop(32),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

train_test_split(image_text, train_text, test_text)
train_dataset = myDataset(img_dir=img_root, img_txt=train_text, transform=transform)
train_dataloader = Data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = myDataset(img_dir=img_root, img_txt=test_text, transform=transform)
test_dataloader = Data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
 
print(len(train_dataset))
print(len(train_dataloader))

print(len(test_dataset))
print(len(test_dataloader))
