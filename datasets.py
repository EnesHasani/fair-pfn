import os
import pandas as pd
import pyarrow.parquet as pq
import torch
from pathlib import Path
from data_generator import DataGenerator
from torch.utils.data import IterableDataset

def save_generated_data(
    Dbias, y_fair,
    filename: Path | str = "data/pre_generated_data.parquet"
):
    if os.path.exists(filename):
        df = pd.read_parquet(filename)
    else:
        df = pd.DataFrame()

    new_df = pd.DataFrame(Dbias.cpu().numpy(), columns=[f"f{i}" for i in range(Dbias.shape[1])])
    new_df["y_fair"] = y_fair.cpu().numpy()

    df = pd.concat([df, new_df], ignore_index=True)
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(filename, index=False)

def pre_generate_data(
    D: int = 100,       # Number of datasets to generate
    U: int = 16,           # Exogenous variables
    H: int = 3,           # MLP depth
    M: int = 16,           # Features
    N: int = 256,         # Samples per dataset
    filename: Path | str = "data/pre_generated_data.parquet",
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    for i in range(D):
        generator = DataGenerator(U=U, H=H, M=M, N=N, device=device)
        Dbias, y_fair = generator.generate_dataset()
        save_generated_data(Dbias, y_fair, filename)
        print(f"Generated dataset {i+1}/{D} saved.")

class SyntheticDataset(IterableDataset):
    def __init__(self, filename: Path | str = "data/pre_generated_data.parquet", batch_size: int = 256):
        self.filename = filename
        self.batch_size = batch_size

    def __iter__(self):
        if not os.path.exists(self.filename):
            raise FileNotFoundError(f"The file {self.filename} does not exist. Please generate data first.")
        reader = pq.ParquetFile(self.filename).iter_batches(batch_size=self.batch_size)

        for batch in reader:
            batch_df = batch.to_pandas()
            features = torch.tensor(batch_df.iloc[:, :-1].values, dtype=torch.float32)
            labels = torch.tensor(batch_df.iloc[:, -1].values, dtype=torch.long)
            yield features, labels

# pre_generate_data(
#     D=100,  # Number of datasets to generate
#     U=16,   # Exogenous variables
#     H=3,    # MLP depth
#     M=16,   # Features
#     N=512,  # Samples per dataset
#     filename="data/pre_generated_data.parquet",
#     device="cuda" if torch.cuda.is_available() else "cpu"
# )
