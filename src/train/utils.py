import torch

"""
from UDA-Animal-Pose (link: )
"""


def mixup(img_src, hm_src, weights_src, img_trg, hm_trg, weights_trg, beta, device):
    m = torch.distributions.beta.Beta(torch.tensor(beta), torch.tensor(beta))
    mix = m.rsample(sample_shape=(img_src.size(0), 1, 1, 1))
    # keep the max value such that the domain labels does not change
    mix = torch.max(mix, 1 - mix)
    mix = mix.to(device)
    img_src_mix = img_src * mix + img_trg * (1. - mix)
    hm_src_mix = hm_src * mix + hm_trg * (1. - mix)
    img_trg_mix = img_trg * mix + img_src * (1. - mix)
    hm_trg_mix = hm_trg * mix + hm_src * (1. - mix)
    weights = torch.max(weights_src, weights_trg)
    return img_src_mix, hm_src_mix, weights, img_trg_mix, hm_trg_mix, weights


# mixup inside domains, mainly mixup different categories to prevent data unbalance
def mixup_withindomain(trg_img, trg_lbl, trg_weights, beta1, beta2, device):
    m = torch.distributions.beta.Beta(torch.tensor(beta1), torch.tensor(beta2))
    mix = m.rsample(sample_shape=(trg_img.size(0), 1, 1, 1))
    mix = mix.to(device)
    index = torch.randperm(trg_img.size(0))
    img_perm = trg_img[index]
    hm_perm = trg_lbl[index]
    weights_perm = trg_weights[index]
    img_mix = trg_img * mix + img_perm * (1 - mix)
    hm_mix = trg_lbl * mix + hm_perm * (1 - mix)
    weights_mix = torch.max(trg_weights, weights_perm)
    return img_mix, hm_mix, weights_mix
