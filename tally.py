import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet50


def calculate_statistics(x):
    mu = x.mean(dim=[2, 3], keepdim=True)
    var = x.var(dim=[2, 3], keepdim=True)
    sig = (var + 1e-6).sqrt()
    norm = (x - mu) / sig

    return norm, mu, sig


class AdaIN(nn.Module):
    def __init__(self, mix_alpha):
        super().__init__()
        self.alpha = mix_alpha

    def mu(self, x):
        """ Takes a (n,c,h,w) tensor as input and returns the average across
        it's spatial dimensions as (h,w) tensor [See eq. 5 of paper]"""
        mu = x.mean(dim=[2, 3], keepdim=True)
        return mu

    def sigma(self, x):
        """ Takes a (n,c,h,w) tensor as input and returns the standard deviation
        across it's spatial dimensions as (h,w) tensor [See eq. 6 of paper] Note
        the permutations are required for broadcasting"""
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + 1e-6).sqrt()
        return sig

    def forward(self, x1, z1, x2, m2, s2):
        mu1 = self.mu(x1)
        sig1 = self.sigma(x1)
        norm1 = (x1 - mu1) / sig1
        bsz = x1.shape[0]
        l = np.random.beta(self.alpha, self.alpha, [bsz, 1])
        l = np.maximum(l, 1 - l)
        l = np.tile(l[..., None, None], (1, *x1.shape[1:]))
        l = torch.tensor(l, dtype=torch.float32).to(x1.device)
        norm_mix = l * norm1 + (1 - l) * z1

        mu2 = self.mu(x2)
        sig2 = self.sigma(x2)
        l = np.random.beta(self.alpha, self.alpha, [bsz, 1])
        l = np.maximum(l, 1 - l)
        l = np.tile(l[..., None, None], (1, *x1.shape[1:]))
        l = torch.tensor(l, dtype=torch.float32).to(x1.device)
        mu_mix = l * mu2 + (1 - l) * m2
        sig_mix = l * sig2 + (1 - l) * s2

        return sig_mix * norm_mix + mu_mix, norm1, sig2, mu2


class TALLY(nn.Module):
    def __init__(self, num_classes, mix_alpha):
        super(TALLY, self).__init__()
        self.num_classes = num_classes
        resnet = resnet50(pretrained=True)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        self.fc = nn.Linear(2048, self.num_classes)

        self.ada = AdaIN(mix_alpha)

    def forward(self, x):
        if self.training:
            return self.forward_train(x)
        else:
            return self.forward_test(x)

    def forward_train(self, x):
        if len(x.shape) == 3:
            x.unsqueeze_(0)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)

        norm, mu, sig = calculate_statistics(x)

        x = self.layer3(x)
        x = self.layer4(x)
        z = self.avgpool(x).squeeze(-1).squeeze(-1)

        return self.fc(z), norm, mu, sig

    def forward_train_aug(self, x1, n1, x2, m2, s2):
        if len(x1.shape) == 3:
            x1.unsqueeze_(0)
        if len(x2.shape) == 3:
            x2.unsqueeze_(0)

        x1 = self.conv1(x1)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        x1 = self.maxpool(x1)

        x1 = self.layer1(x1)
        x1 = self.layer2(x1)

        x2 = self.conv1(x2)
        x2 = self.bn1(x2)
        x2 = self.relu(x2)
        x2 = self.maxpool(x2)

        x2 = self.layer1(x2)
        x2 = self.layer2(x2)

        x, norm, sig, mu = self.ada(x1, n1, x2, m2, s2)

        x = self.layer3(x)
        x = self.layer4(x)
        z = self.avgpool(x).squeeze(-1).squeeze(-1)

        return self.fc(z), norm, sig, mu

    def forward_test(self, x):
        if len(x.shape) == 3:
            x.unsqueeze_(0)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        z = self.avgpool(x).squeeze(-1).squeeze(-1)

        return self.fc(z)

