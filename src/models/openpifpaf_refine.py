# import openpifpaf
import torch.nn as nn

from src.models.refinenet import RefineNet


class OpenPifPafRefine(nn.Module):
    def __init__(self, backbone, head, refinenet):
        super(OpenPifPafRefine, self).__init__()
        self.backbone = backbone
        self.head = head
        self.refinenet = refinenet

    def forward(self, x):
        x = self.backbone(x)
        x, features = self.head(x)
        x_refine = self.refinenet(features)

        return x, x_refine


def get_openpifpaf_model(checkpoint: str):
    net_cpu, _ = openpifpaf.network.Factory(checkpoint=checkpoint).factory()
    return net_cpu


def openpifpaf_refine(**kwargs):
    net = get_openpifpaf_model(kwargs['openpifpaf_checkpoint'])
    refinenet = RefineNet(256, (64, 64), kwargs['num_classes'])
    posenet = OpenPifPafRefine(net.base_net, net.head_nets, refinenet)
    return posenet
