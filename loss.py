'''
This is the file to define the loss function for VAE model.

@Author: Xiangyun Rao
@Date: 2023.3.13
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


# Reconstruction + KL divergence losses summed over all elements and batch
class KLDLoss(nn.Module):
    def __init__(self):
        super(KLDLoss, self).__init__()
        self.kld_loss = nn.KLDivLoss(reduction='batchmean')
    def forward(self, mean, logvar):
        kld_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        return kld_loss
    
class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()
        self.kld_loss = KLDLoss()
        self.bce_loss = nn.BCELoss(reduction='sum')
    def forward(self, recon_x, x, mean, logvar):
        bce_loss = self.bce_loss(recon_x, x)
        kld_loss = self.kld_loss(mean, logvar)
        return bce_loss + kld_loss