import torch
from torch.autograd import forward_ad
import torch.nn as nn
import torchvision

class Generator(nn.Module):
    def __init__(self, latent_dim=10, out_shape=(64, 64)):
        super().__init__()

        self.latent_dim = latent_dim
        self.out_shape = out_shape

        def hidden(in_channels, out_channels, k=3, p=1, out=False):
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, k, 1, p),
                nn.LeakyReLU(0.1) if not out else nn.Tanh(),
                nn.BatchNorm2d(out_channels)
                )

        self.hidden1 = hidden(2, 4, 9, 0)
        self.hidden2 = hidden(4, 6, 9, p=0)
        self.hidden3 = hidden(6, 8, 7, p=0)
        self.hidden4 = hidden(8, 16, 7, p=0)
        self.hidden5 = hidden(16, 24, 7, p=0)
        self.hidden6 = hidden(25, 32, 5, p=0)
        self.hidden7 = hidden(32, 32, 5, p=0)
        self.hidden8 = hidden(32, 32, 5, p=0)
        self.hidden9 = hidden(33, 32, 5, p=0)
        self.hidden10 = hidden(32, 3, 5, p=0, out=True)

    def forward(self, x, label):
        c = torchvision.transforms.Resize((self.latent_dim, self.latent_dim))(label)
        c = c.view(-1, self.latent_dim * self.latent_dim)

        # Put embedded label to the last channel
        x = torch.cat([x, c], 1).view(-1, 2, self.latent_dim, self.latent_dim)

        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.hidden4(x)
        x = self.hidden5(x)
        x = torch.cat([x, torchvision.transforms.Resize(x.shape[2:])(label)], dim=1)
        x = self.hidden6(x)
        x = self.hidden7(x)
        x = self.hidden8(x)
        x = torch.cat([x, torchvision.transforms.Resize(x.shape[2:])(label)], dim=1)
        x = self.hidden9(x)
        x = self.hidden10(x)

        return x

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        def hidden(in_channels, out_channels, k=3, p=1, out=False):
            return nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, k, padding=p),
                    nn.Dropout(0.3),
                    nn.BatchNorm2d(out_channels),
                    nn.LeakyReLU(0.1)
                    )

        self.label_emb = hidden(1, 1, 5, 2)

        self.hidden1 = hidden(4, 8, p=0)
        self.hidden2 = hidden(8, 16, p=0)
        self.hidden3 = hidden(16, 32, p=0)

        self.out = nn.LazyLinear(1)

    def forward(self, x, label):
        label = self.label_emb(label)

        # Put embedded label to the last channel
        x = torch.cat([x, label], 1)

        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)

        return torch.sigmoid(self.out(torch.flatten(x, 1)))
