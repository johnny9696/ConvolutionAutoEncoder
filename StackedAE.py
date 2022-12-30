import torch
import torch.nn as nn
from torch import Tensor


class Encoder(nn.Module):
    def __init__(self, encoder_dim=1,
    hidden_1dim=3,
    hidden_2dim=5,
    kernel=5):
        super().__init__()
        self.encoder_dim=encoder_dim
        self.hidden_dim_1=hidden_1dim
        self.hidden_dim_2=hidden_2dim
        self.kernel=kernel

        self.conv2d_layer_1=nn.Conv2d(self.encoder_dim,self.hidden_dim_1,kernel_size=kernel)
        self.conv2d_layer_2=nn.Conv2d(self.hidden_dim_1,self.hidden_dim_2,kernel_size=kernel)
        self.relu=nn.ReLU()

    def forward(self,mel):
        x=self.conv2d_layer_1(mel)
        x=self.relu(x)
        x=self.conv2d_layer_2(x)
        x=self.relu(x)

        return x

class Decoder(nn.Module):
    def __init__(self,
    encoder_dim,
    hidden_1dim,
    hidden_2dim,
    kernel=5):
    
        super().__init__()
        self.Tconv2d_layer1=nn.ConvTranspose2d(hidden_2dim, hidden_1dim, kernel_size=kernel)
        self.Tconv2d_layer2=nn.ConvTranspose2d(hidden_1dim, encoder_dim, kernel_size=kernel)
        self.relu=nn.ReLU()

    def forward(self,z):
        z=self.Tconv2d_layer1(z)
        z=self.relu(z)
        z=self.Tconv2d_layer2(z)
        z=self.relu(z)

        return z


class Convolution_Auto_Encoder(nn.Module):
    def __init__(self,
    encoder_dim,
    hidden_1dim,
    hidden_2dim,
    kernel=5):
        super().__init__()
        self.encoder_dim=encoder_dim
        self.hidden_dim_1=hidden_1dim
        self.hidden_dim_2=hidden_2dim

        #convolution autoencoder
        self.encoder=Encoder(encoder_dim=self.encoder_dim, hidden_1dim=self.hidden_dim_1, hidden_2dim=self.hidden_dim_2,kernel=kernel)
        self.decoder=Decoder(encoder_dim=self.encoder_dim, hidden_1dim=self.hidden_dim_1, hidden_2dim=self.hidden_dim_2,kernel=kernel)

    def forward(self,mel,classification=False):
        mel=self.encoder(mel)
        if classification == False:
            mel=self.decoder(mel)
        return mel
    
    def get_vector(self,mel):
        return self.encoder(mel)

class classification(nn.Module) : 
    def __init__(self, 
    hidden_2dim,
    height,
    width,
    output_dim = 1,
    hidden_channel = 256,
    output_channel = None
    ):
        super().__init__()

        self.conv1x1 = nn.Conv2d(hidden_2dim , output_dim, kernel_size = 1)
        self.linear1 = nn.Linear(height * width, hidden_channel)
        self.linear2 = nn.Linear(hidden_channel, output_channel)

    def forward(self, x) :
        x = self.conv1x1(x)
        x = self.linear1(x)
        x = self.linear2(x)
        return x
    
    def get_vector(self, x):
        x = self.conv1x1(x)
        x = self.linear1(x)
        return x

class Convolution_AE_Classification(nn.Module):
    def __init__(self,
    encoder_dim,
    hidden_1dim,
    hidden_2dim,
    hps,
    kernel = 5,
    n_mels = 80,
    ):
        super().__init__()
        self.encoder_dim=encoder_dim
        self.hidden_dim_1=hidden_1dim
        self.hidden_dim_2=hidden_2dim
        self.mels = n_mels

        width = (hps.data.n_mel_channels-(kernel-1))-(kernel-1)
        height = int(hps.data.sampling_rate / hps.data.win_length * (hps.data.win_length / hps.data.hop_length) * hps.data.slice_length)
        height = height -2- 2*kernel
    

        #convolution autoencoder
        self.encoder=Encoder(encoder_dim=self.encoder_dim, hidden_1dim=self.hidden_dim_1, hidden_2dim=self.hidden_dim_2,kernel=kernel)
        self.classification= classification(hidden_2dim= self. hidden_dim_2, 
        output_channel= hps.model.output_channel, height=height, width=width)
        self.log_softmax= torch.nn.LogSoftmax()

    def forward(self, x, no_grad =True) :
        if no_grad:
            with torch.no_grad:
                x = self.encoder(x)
        x = self.classification(x)
        x = self.log_softmax(x)
        return x
    
    def get_vector(self, x):
        x = self.encoder(x)
        x = self.classification.get_vector(x)
        x = x.squeeze()
        return x
