import torch
import torch.nn as nn
from pathlib import Path
from tabpfn.base import load_model_criterion_config


class FairPFNModel(nn.Module):
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__()
        self.tabpfn, _, _ = load_model_criterion_config(
            model_path=None,
            check_bar_distribution_criterion=False,
            cache_trainset_representation=False,
            which="classifier",
            version="v2",
            download=True
        )
        self.device = device
        self.tabpfn = self.tabpfn.to(self.device)
        for param in self.tabpfn.parameters():
            param.requires_grad = True

    def forward(self, **forward_kwargs):
        return self.tabpfn(**forward_kwargs)

    def save_model(self, path = Path("models/fairpfn_model.pth")):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.tabpfn.state_dict(), path)

    def load_model(self, path, eval_mode=False):
        if isinstance(path, str):
            path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"The model file {path} does not exist.")
        self.tabpfn.load_state_dict(torch.load(path, map_location=self.device))
        if eval_mode:
            self.tabpfn.eval()
            for param in self.tabpfn.parameters():
                param.requires_grad = False
        return self
