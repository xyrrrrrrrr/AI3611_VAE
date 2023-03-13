'''
This is a file to check whether here is MNIST dataset or not and download it if not 
and construct a dataloader for training and validation.

@Author: Xiangyun Rao
@Date: 2023.3.13
'''

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

# Download MNIST Dataset
train_dataset = datasets.MNIST(root='./data/', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='./data/', train=False, transform=transforms.ToTensor(), download=False)

# MNist Data Loader
batch_size=50
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# iterator over train set
counter = 0
for batch_idx, (data, _) in enumerate(train_loader):
    print ("In train stage: data size: {}".format(data.size()))
    if batch_idx == 0:
        nelem = data.size(0)
        nrow  = 10
        save_image(data.view(nelem, 1, 28, 28), './images/image_0' + str(counter) + '.png', nrow=nrow)
        counter += 1

# iterator over test set
for data, _ in test_loader:
    print ("In test stage: data size: {}".format(data.size()))




