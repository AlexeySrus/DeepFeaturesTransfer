import torch
import torch.nn as nn
from torchvision.models.resnet import resnext50_32x4d


class RGB2RGBAutoencoder(nn.Module):
    def __init__(self, extractor_model=resnext50_32x4d(False)):
        super(RGB2RGBAutoencoder, self).__init__()

        self.encoder = torch.nn.Sequential(
            *(list(extractor_model.children())[:-2])
        )

        self.decoder = torch.nn.Sequential(
            nn.ConvTranspose2d(2048, 512, 5, stride=1, padding=0, bias=False),
            nn.ConvTranspose2d(512, 256, 5, stride=2, padding=0, bias=False),
            nn.ConvTranspose2d(256, 128, 5, stride=2, padding=0, bias=False),
            nn.ConvTranspose2d(128, 64, 5, stride=2, padding=0, bias=False),
            nn.ConvTranspose2d(64, 32, 5, stride=2, padding=0, bias=False),
            nn.ConvTranspose2d(32, 3, 4, stride=1, padding=0, bias=False),
        )

    def forward(self, x):
        features = self.encoder(x)
        decoded_x = self.decoder(features)
        return decoded_x, features

    def decode(self, f):
        return self.decoder(f)


class Edge2EdgeAutoencoder(nn.Module):
    def __init__(self, extractor_model=resnext50_32x4d(False)):
        super(Edge2EdgeAutoencoder, self).__init__()

        self.encoder = torch.nn.Sequential(
            nn.Conv2d(1, 3, 1, bias=False),
            *(list(extractor_model.children())[:-2])
        )

        self.decoder = torch.nn.Sequential(
            nn.ConvTranspose2d(2048, 512, 5, stride=1, padding=0, bias=False),
            nn.ConvTranspose2d(512, 256, 5, stride=2, padding=0, bias=False),
            nn.ConvTranspose2d(256, 128, 5, stride=2, padding=0, bias=False),
            nn.ConvTranspose2d(128, 64, 5, stride=2, padding=0, bias=False),
            nn.ConvTranspose2d(64, 32, 5, stride=2, padding=0, bias=False),
            nn.ConvTranspose2d(32, 1, 4, stride=1, padding=0, bias=False),
        )

    def forward(self, x):
        features = self.encoder(x)
        decoded_x = self.decoder(features)
        return decoded_x, features

    def decode(self, f):
        return self.decoder(f)
