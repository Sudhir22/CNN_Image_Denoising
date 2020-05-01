import torch
import torch.nn as nn



class Encoder(nn.Module):
    """Encoder network to map from an RGB image to a latent feature vector."""

    def __init__(self, z_dim=64, img_size=64):
        super(Encoder, self).__init__()

        self.z_dim = z_dim
        self.hidden_layer1 = nn.Sequential(nn.Linear(img_size*img_size*3,z_dim*1*1),nn.BatchNorm1d(z_dim*1*1),nn.Sigmoid())
        self.output_layer = nn.Sequential(nn.Linear(z_dim*1*1,z_dim*1*1),nn.Sigmoid())


    def forward(self, x):
        x = x.view(x.size()[0], -1)
        x = self.output_layer(self.hidden_layer1(x))
        return x


class Decoder(nn.Module):
    """Decoder network to map from a latent feature vector to an RGB image."""

    def __init__(self, z_dim=64, img_size=64):
        super(Decoder, self).__init__()

        assert img_size==64
        self.z_dim = z_dim
        self.layer1 = nn.Sequential(nn.ConvTranspose2d(z_dim*1*1,128,7,1),nn.BatchNorm2d(128),nn.ReLU(),nn.Conv2d(128,128,3,1,1), nn.BatchNorm2d(128),nn.Sigmoid())
        self.layer2 = nn.Sequential(nn.ConvTranspose2d(128,64,3,2),nn.BatchNorm2d(64),nn.ReLU(),nn.Conv2d(64,64,3,1,1), nn.BatchNorm2d(64),nn.Sigmoid())
        self.layer3 = nn.Sequential(nn.ConvTranspose2d(64,64,3,2),nn.BatchNorm2d(64),nn.ReLU(),nn.Conv2d(64,64,3,1,1), nn.BatchNorm2d(64),nn.Sigmoid())
        self.layer4 = nn.Sequential(nn.ConvTranspose2d(64,32,3,2),nn.BatchNorm2d(32),nn.ReLU(),nn.Conv2d(32,32,3,1,1), nn.BatchNorm2d(32),nn.Sigmoid())
        self.layer5 = nn.Sequential(nn.ConvTranspose2d(32,32,2,1),nn.BatchNorm2d(32),nn.ReLU(),nn.Conv2d(32,3,1,1), nn.Sigmoid())


    def forward(self, x):
        x = x.view(x.size()[0], self.z_dim, 1, 1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x
