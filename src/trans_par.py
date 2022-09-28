import torch
import torch.nn as nn


# from TransPar repository
def LTH(model, keep_ratio=0.5, eps=1e-10):
    grads = dict()
    modules = list(model.modules())

    for idx, layer in enumerate(model.modules()):
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            if layer.weight.grad is None:  # There are exists a final linear layer of Reset is fixed, thus we pass it.
                continue
            else:
                grads[modules[idx]] = torch.abs(layer.weight.data * layer.weight.grad)  # -theta_q Hg

    all_scores = torch.cat([torch.flatten(x) for x in grads.values()])
    norm_factor = torch.abs(torch.sum(all_scores)) + eps
    # print("** norm factor:", norm_factor)
    all_scores.div_(norm_factor)

    num_params_to_rm = int(len(all_scores) * (1 - keep_ratio))
    threshold, _ = torch.topk(all_scores, num_params_to_rm, sorted=True)
    # import pdb; pdb.set_trace()
    acceptable_score = threshold[-1]
    # print(all_scores)
    # print('** accept: ', acceptable_score)
    keep_masks = dict()
    for m, g in grads.items():
        # keep_masks[m] = (g >= acceptable_score).float()
        keep_masks[m] = ((g / norm_factor) >= acceptable_score).float()

    # print(torch.sum(torch.cat([torch.flatten(x == 1) for x in keep_masks.values()])))
    # print(keep_masks.keys())
    # TODO register mask
    for m in keep_masks.keys():
        if isinstance(m, nn.Conv2d) or isinstance(layer, nn.Linear):
            mask = keep_masks[m]
            # print(m, mask)
            m.weight.grad.mul_(mask)
            m.weight.grad.mul_(mask)
            m.weight.grad.mul_(mask)
