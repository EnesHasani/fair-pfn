import torch
from torch.utils.data import Dataset
from typing import Tuple, List


class FairPFNDataset(Dataset):
    def __init__(self, Dbias: List[Tuple[int, List[float], int]], Dfair: List[int]):
        self.A = torch.tensor([a for a, _, _ in Dbias], dtype=torch.float32).unsqueeze(1)
        self.X = torch.tensor([x for _, x, _ in Dbias], dtype=torch.float32)
        self.y_biased = torch.tensor([y for _, _, y in Dbias], dtype=torch.float32).unsqueeze(1)
        self.y_fair = torch.tensor(Dfair, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.cat((self.A[idx], self.X[idx]), dim=0), self.y_fair[idx]  # (features, label)