import torch
import torch.nn as nn


class BottleneckRefinenet(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 2, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 2)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes * 2,
                      kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * 2),
        )
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class RefineNet(nn.Module):
    def __init__(self, lateral_channel, out_shape, num_class, dual_branch=False):
        super().__init__()
        cascade = []
        num_cascade = 4
        for i in range(num_cascade):
            cascade.append(self._make_layer(lateral_channel, num_cascade - i - 1, out_shape))
        self.cascade = nn.ModuleList(cascade)
        self.final_predict = self._predict(4 * lateral_channel, num_class)
        self.dual_branch = dual_branch
        if dual_branch:
            self.final_predict_2 = self._predict(4 * lateral_channel, num_class)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, mean=0, std=0.001)

    @classmethod
    def _make_layer(cls, input_channel, num, output_shape):
        layers = list()
        for i in range(num):
            layers.append(BottleneckRefinenet(input_channel, 128))
        layers.append(nn.Upsample(size=output_shape, mode='bilinear', align_corners=True))
        return nn.Sequential(*layers)

    @classmethod
    def _predict(cls, input_channel, num_class, with_bias_end=True):
        layers = list()
        layers.append(BottleneckRefinenet(input_channel, 128))
        layers.append(nn.Conv2d(256, num_class, kernel_size=3, stride=1, padding=1, bias=with_bias_end))
        return nn.Sequential(*layers)

    def forward(self, x):
        refine_fms = []
        for i in range(4):
            refine_fms.append(self.cascade[i](x[i]))
        out = torch.cat(refine_fms, dim=1)
        out1 = self.final_predict(out)
        if self.dual_branch:
            out2 = self.final_predict_2(out)
            return [out1, out2], out
        else:
            return out1, out
