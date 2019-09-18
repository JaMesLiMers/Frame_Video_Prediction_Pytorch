import torch


class L1_L2_Loss(torch.nn.Module):
    def __init__(self):
        super(L1_L2_Loss, self).__init__()
    
    def forward(self, target, pred):
        diff = target - pred
        loss_ = torch.pow(diff, 2) + torch.abs(diff) # L2 + L1
        return loss_.mean()
