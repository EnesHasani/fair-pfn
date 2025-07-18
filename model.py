import torch.nn as nn
from tabpfn.base import load_model_criterion_config


class FairPFNModel(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.tabpfn, _, _ = load_model_criterion_config(
            model_path=None,
            check_bar_distribution_criterion=False,
            cache_trainset_representation=False,
            which="classifier",
            version="v2",
            download=True
        )
        self.tabpfn = self.tabpfn.to(device)
        for param in self.tabpfn.parameters():
            param.requires_grad = True

    def forward(self, **forward_kwargs):
        return self.tabpfn(**forward_kwargs)
