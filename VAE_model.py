'''
This file contains my implementation of VAE model.

@Author: Xiangyun Rao
@Date: 2023.3.13
'''
import torch
import torch.nn as nn

z_dim = 1 # That is the dimension of the latent space

class Downsample(nn.Module):
    def __init__(self, input_size, output_size):
        super(Downsample, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.Conv = nn.Conv2d(self.input_size, self.output_size, 3, 1, 1)

    def forward(self, input):
        output = self.Conv(input)
        return output



class Upsample(nn.Module):
    def __init__(self, input_size, output_size):
        super(Upsample, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.Conv = nn.Conv2d(self.input_size, self.output_size, 3, 1, 1)

    def forward(self, input):
        output = self.Conv(input)
        return output  

class Encoder(nn.Module):
    def __init__(self, input_size=1, output_size=1, hidden_size=z_dim, dropout=0.1):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)
        self.Downsample1 = Downsample(self.input_size, self.output_size)
        self.Downsample2 = Downsample(self.output_size, self.output_size)
        self.Downsample3 = Downsample(self.output_size, self.output_size)
        self.mean = nn.Linear(28*28, self.hidden_size)
        self.variance = nn.Linear(28*28, self.hidden_size)

    def forward(self, input):
        output = self.Downsample1(input)
        output = self.dropout(output)
        output = self.Downsample2(output)
        output = self.dropout(output)
        output = self.Downsample3(output)
        output = nn.Flatten()(output)
        mean = self.mean(output)
        variance = self.variance(output)
        return mean, variance
    
    def reparameterize(self, mean, std):
        eps = torch.randn_like(std)
        return mean + eps * std


class Decoder(nn.Module):
    # define a generator
    def __init__(self, input_size=z_dim, output_size=1):
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.fc = nn.Linear(self.input_size, 28*28)
        self.Upsample1 = Upsample(self.input_size, self.output_size)
        self.Upsample2 = Upsample(self.output_size, self.output_size)
        self.Upsample3 = Upsample(self.output_size, self.output_size)


    def forward(self, input):
        output = self.fc(input)
        output = output.view(-1, 1, 28, 28)
        output = self.Upsample1(output)
        output = self.Upsample2(output)
        output = self.Upsample3(output)
        # change to 0 or 1
        output = torch.sigmoid(output)
        return output

        
class VAE(nn.Module):
    def __init__(self, input_size=1, output_size=1, hidden_size=z_dim, dropout=0.1):
        super(VAE, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.encoder = Encoder(self.input_size, self.output_size, self.hidden_size, dropout)
        self.decoder = Decoder(self.hidden_size, self.input_size)

    def forward(self, input):
        mean, var = self.encoder(input)
        self.mean = mean
        self.variance = var
        z = self.encoder.reparameterize(mean, var)
        output = self.decoder(z)
        return output
    
    def _get_mu(self):
        return self.mean

    def _get_var(self):
        return self.variance