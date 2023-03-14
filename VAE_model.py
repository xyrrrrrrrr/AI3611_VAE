'''
This file contains my implementation of VAE model.

@Author: Xiangyun Rao
@Date: 2023.3.13
'''
import torch
import torch.nn as nn

z_dim = 2 # That is the dimension of the latent space

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
        self.fc = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256, 1024)
        self.mean = nn.Linear(1024, self.hidden_size)
        self.logvar = nn.Linear(1024, self.hidden_size)

    def forward(self, input):
        output = nn.Flatten()(input)
        output = self.fc(output)
        output = torch.relu(output) 
        output = self.fc2(output)
        output = torch.relu(output) 
        mean = self.mean(output)
        logvar = self.logvar(output)
        return mean, logvar
    
    def reparameterize(self, mean, logvar):
        eps = torch.randn_like(logvar)
        return mean + eps * torch.exp(logvar)


class Decoder(nn.Module):
    # define a generator
    def __init__(self, input_size=z_dim, output_size=1):
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.fc1 = nn.Linear(self.input_size, 28*28)
        self.fc2 = nn.Linear(28*28, 28*28)

    def forward(self, input):
        output = self.fc1(input)
        output = torch.relu(output) 
        output = self.fc2(output)
        output = output.view(-1, 1, 28, 28)
        output = torch.sigmoid(output)
        return output

        
class VAE(nn.Module):
    def __init__(self, input_size=1, output_size=1, hidden_size=z_dim, dropout=0):
        super(VAE, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.encoder = Encoder(self.input_size, self.output_size, self.hidden_size, dropout)
        self.decoder = Decoder(self.hidden_size, self.input_size)

    def forward(self, input):
        mean, logvar = self.encoder(input)
        self.mean = mean
        self.logvar = logvar
        z = self.encoder.reparameterize(mean, logvar)
        output = self.decoder(z)
        return output
    
    def _get_mu(self):
        return self.mean

    def _get_var(self):
        return self.logvar