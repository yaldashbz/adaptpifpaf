import torch.nn as nn


def detach_model(model: nn.Module):
    for param in model.parameters():
        param.detach()
