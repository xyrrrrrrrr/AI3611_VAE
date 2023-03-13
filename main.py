'''
This is the file to train model and generate results.

@Author: Xiangyun Rao
@Date: 2023.3.13
'''
import os
import cv2
import time
import torch
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
    plt.plot(losses, label=label)
    # save model
    cur = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    torch.save(VAE_model.state_dict(), os.path.join(save_dir, 'VAE_model_' + cur + '.pth'))
else:
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    # load model
    VAE_model.load_state_dict(torch.load(args.model))

    # test model and save results
    for batch_idx, (data, _) in enumerate(test_loader):
        data = data.to(device)
        # generate results
        recon_x = VAE_model(data)
        # save results
        # merge two images
        img = torch.cat((data, recon_x), dim=0)
        # save image
        cur = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        cv2.imwrite(os.path.join(results_dir, 'results_' + cur + '.jpg'), img)


    
