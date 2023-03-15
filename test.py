'''
This is the file to train model and generate results.

@Author: Xiangyun Rao
@Date: 2023.3.13
'''
import os
import cv2
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim import Adam, SGD
from args import get_args
from VAE_model import VAE
from loss import MyLoss

# get arguments
args = get_args()
device = torch.device('cuda:{}'.format(args.device) if args.device != '-1' else 'cpu')
lr = args.lr
batch_size = args.batch_size
epochs = args.epochs
save_dir = args.save_dir
results_dir = args.results_dir
testing = args.testing
model_name = args.model


# load data

train_dataset = datasets.MNIST(root='./data/', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='./data/', train=False, transform=transforms.ToTensor(), download=False)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# define model
VAE_model = VAE().to(device)
loss_function = MyLoss().to(device)
if not testing:
    # train model and save model
    if args.optimizer == 'Adam':
        optimizer = Adam(lr=lr, params=VAE_model.parameters())
    elif args.optimizer == 'SGD':
        optimizer = SGD(lr=lr, momentum=0.9, params=VAE_model.parameters())
    # check whether the directory exists
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    losses = []
    for epoch in range(1, epochs + 1):
        if args.lr_scheduled:
            if epoch % 10 == 0:
                lr = lr * 0.1
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            # forward
            recon_x = VAE_model(data)
            mu = VAE_model._get_mu()
            var = VAE_model._get_var()
            # backward
            loss = loss_function(recon_x, data, mu, var)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset),
                    100 * batch_idx / len(train_loader), 
                    loss.item()), end='\r', flush=True)
            if batch_idx % 100 == 0:
                losses.append(loss.item())
    # plot loss
    label = 'optimizer: {}, lr: {}, batch_size: {}, epochs: {}'.format(args.optimizer, lr, batch_size, epochs)
    cur = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    plt.plot(losses, label=label)
    plt.savefig(os.path.join(save_dir, 'loss'+ cur +'.png'))
    # save model
    
    torch.save(VAE_model.state_dict(), os.path.join(save_dir, 'VAE_model_' + cur + '.pth'))
else:
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    # load model
    VAE_model.load_state_dict(torch.load(args.model))

    # test model and save results
    counter = 0
    img_list = []
    z_list = []
    for batch_idx, (data, _) in enumerate(test_loader):
        if counter >= 100:
            break
        data = data.to(device)
        # generate results
        recon_x = VAE_model(data)
        # change to 0 or 255
        img = torch.where(recon_x > 0.5, torch.ones_like(recon_x), torch.zeros_like(recon_x))
        z = VAE_model._get_z()
        # save image
        img = img.cpu().detach().numpy()
        z = z.cpu().detach().numpy()
        for i in range(len(img)):
            img_i = img[i][0, :, :] * 255
            img_i = img_i.astype(np.uint8)
            if abs(z[i][0]) <= 5 and abs(z[i][1]) <= 5:
                img_list.append(img_i)
                z_list.append(z[i])
            counter += 1
            if counter >= 100:
                break

    # 按照z(2维)的值进行图片二维排序，然后reshape成8*8的图片
    z_list = np.array(z_list)
    img_list = np.array(img_list)
    z_list = z_list[:, 0] + z_list[:, 1] * 1j
    z_list = np.argsort(z_list)
    img_list = img_list[z_list]
    img_list = img_list.reshape(10, 10, 28, 28)
    # 按照8*8的顺序进行拼接
    background = np.zeros((10 * 28, 10 * 28))
    for i in range(10):
        for j in range(10):
            background[i * 28: (i + 1) * 28, j * 28: (j + 1) * 28] = img_list[i][j]

    cv2.imwrite(os.path.join('results.png'), background)


    
