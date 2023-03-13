'''
This file contains my implementation of VAE model.

@Author: Xiangyun Rao
@Date: 2023.3.13
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils import clip_grad_norm_
from torch.nn import init
import numpy as np
import math

z_dim = 100 # That is the dimension of the latent space

class Downsample(nn.modules):
    def __init__(self, input_size, output_size):
        super(Downsample, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.Conv = nn.Conv2d(self.input_size, self.output_size, 3, 1, 1)

    def forward(self, input):
        output = self.Conv(input)
        return output



class Upsample(nn.modules):
    def __init__(self, input_size, output_size):
        super(Upsample, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.Conv = nn.Conv2d(self.input_size, self.output_size, 3, 1, 1)

    def forward(self, input):
        output = self.Conv(input)
        return output  

class Encoder(nn.modules):
    def __init__(self, input_size=28, hidden_size=z_dim, num_layers=100, dropout=0.2):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.Downsample1 = Downsample(self.input_size, self.hidden_size*4)
        self.Downsample2 = Downsample(self.hidden_size*4, self.hidden_size*2)
        self.Downsample3 = Downsample(self.hidden_size*2, self.hidden_size)

    def forward(self, input):
        output = self.Downsample1(input)
        output = self.Downsample2(output)
        output = self.Downsample3(output)
        return output


class Decoder(nn.modules):
    # define a generator
    def __init__(self, input_size=z_dim, output_size=28):
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.fc = nn.Linear(self.input_size, self.output_size)
        self.Upsample1 = Upsample(self.input_size, self.input_size*2)
        self.Upsample2 = Upsample(self.input_size*2, self.input_size*4)
        self.Upsample3 = Upsample(self.input_size*4, self.output_size)


    def forward(self, input):
        output = self.fc(input)
        output = self.Upsample1(output)
        output = self.Upsample2(output)
        output = self.Upsample3(output)
        return output

        
