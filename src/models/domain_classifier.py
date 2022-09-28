import torch.nn as nn


class DomainClassifier(nn.Module):
    def __init__(self, input_dim=1024, ndf=64, with_bias=False):
        super(DomainClassifier, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, ndf, kernel_size=4, stride=2, padding=1, bias=with_bias)
        self.conv2 = nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=with_bias)
        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=with_bias)
        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1, bias=with_bias)
        self.conv5 = nn.Conv2d(ndf * 8, ndf * 16, kernel_size=4, stride=2, padding=1, bias=with_bias)
        self.conv6 = nn.Conv2d(ndf * 16, 1, kernel_size=2, stride=1, bias=with_bias)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.conv4(x)
        x = self.leaky_relu(x)
        x = self.conv5(x)
        x = self.leaky_relu(x)
        x = self.conv6(x)
        return x
