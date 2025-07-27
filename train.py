import torch
import wandb
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torch.utils.data import DataLoader
from model import FairPFNModel
from data_generator import DataGenerator   
from datasets import SyntheticDataset

def pretrain_fairpfn_on_pre_generated_data(
    E: int,               # Epochs
    batch_size: int = 256,
    filename: Path | str = "data/generated_data.parquet",
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    model = FairPFNModel(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    dataset = SyntheticDataset(filename=filename, batch_size=batch_size)
    dataloader = DataLoader(dataset, batch_size=None)

    wandb.define_metric("dataset")
    wandb.define_metric("loss_step", step_metric="dataset")
    wandb.define_metric("accuracy_step", step_metric="dataset")

    wandb.define_metric("epoch")
    wandb.define_metric("loss_epoch", step_metric="epoch")
    wandb.define_metric("accuracy_epoch", step_metric="epoch")

    number_of_datasets = 0
    for epoch in range(E):
        print(f"Epoch {epoch+1}/{E}")
        loss_epoch = 0
        accuracy_epoch = 0
        for step, (Dbias, y_fair) in enumerate(dataloader):
            Dbias, y_fair = Dbias.to(device), y_fair.to(device)

            split = int(0.7 * len(Dbias))

            train_biased_features = Dbias[:split, :-1].unsqueeze(1).to(device)
            train_biased_labels = Dbias[:split, -1].unsqueeze(1).to(device)
            val_biased_features = Dbias[split:, :-1].unsqueeze(1).to(device)
            num_classes = len(torch.unique(train_biased_labels))
            num_classes = len(torch.unique(train_biased_labels))
            if num_classes < 2:
                print("Skipping batch due to only one class present in training labels.")
                continue

            model.train()
            optimizer.zero_grad()
            forward_kwargs = dict(
                train_x = train_biased_features,
                train_y = train_biased_labels,
                test_x = val_biased_features,
                categorical_inds=None,
            )

            pred_fair_logits = model(**forward_kwargs)
            pred_fair_logits = pred_fair_logits[:, :, :num_classes]
            pred_fair_logits = pred_fair_logits.reshape(-1, pred_fair_logits.shape[-1])
            loss = criterion(pred_fair_logits, y_fair[split:])
            accuracy = (pred_fair_logits.argmax(dim=1) == y_fair[split:]).float().mean()

            loss.backward()
            optimizer.step()
            loss_epoch += loss.item()
            accuracy_epoch += accuracy.item()

            wandb.log({
                "loss_step": loss.item(),
                "dataset": epoch * number_of_datasets + step + 1,
            })
            wandb.log({
                "accuracy_step": accuracy.item(),
                "dataset": epoch * number_of_datasets + step + 1,
            })

        number_of_datasets = step + 1
        wandb.log({
            "loss_epoch": loss_epoch / number_of_datasets,
            "epoch": epoch
        })
        wandb.log({
            "accuracy_epoch": accuracy_epoch / number_of_datasets,
            "epoch": epoch
        })

        if epoch % 50 == 0 and epoch > 0:
            model.save_model(Path(f"models/fairpfn_model_epoch_{epoch + 1}.pth"))
            print(f"Model saved at epoch {epoch + 1}")

    return model

def pretrain_fairpfn(
    E: int,               # Epochs
    S: int,               # Steps per epoch
    U: int = 16,           # Exogenous variables
    H: int = 3,           # MLP depth
    M: int = 16,           # Features
    N: int = 256,         # Samples per dataset
    device="cuda" if torch.cuda.is_available() else "cpu"
):

    model = FairPFNModel(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    wandb.define_metric("dataset")
    wandb.define_metric("loss_step", step_metric="dataset")
    wandb.define_metric("accuracy_step", step_metric="dataset")

    wandb.define_metric("datasets_trained_times_100")
    wandb.define_metric("datasets_loss", step_metric="datasets_trained_times_100")
    wandb.define_metric("datasets_accuracy", step_metric="datasets_trained_times_100")

    wandb.define_metric("epoch")
    wandb.define_metric("epoch_loss", step_metric="epoch")
    wandb.define_metric("epoch_accuracy", step_metric="epoch")

    for epoch in range(E):
        print(f"Epoch {epoch+1}/{E}")
        datasets_loss, datasets_accuracy, epoch_loss, epoch_accuracy = 0, 0, 0, 0
        for step in range(S):
            generator = DataGenerator(U=U, H=H, M=M, N=N, device=device)
            Dbias, y_fair = generator.generate_dataset()

            split = int(0.7 * len(Dbias))

            train_biased_features = Dbias[:split, :-1].unsqueeze(1).to(device)
            train_biased_labels = Dbias[:split, -1].unsqueeze(1).to(device)
            val_biased_features = Dbias[split:, :-1].unsqueeze(1).to(device)
            num_classes = len(torch.unique(train_biased_labels))
            num_classes = len(torch.unique(train_biased_labels))
            if num_classes < 2:
                print("Skipping batch due to only one class present in training labels.")
                continue

            model.train()
            optimizer.zero_grad()
            forward_kwargs = dict(
                train_x = train_biased_features,
                train_y = train_biased_labels,
                test_x = val_biased_features,
                categorical_inds=None,
            )

            pred_fair_logits = model(**forward_kwargs)
            pred_fair_logits = pred_fair_logits[:, :, :num_classes]
            pred_fair_logits = pred_fair_logits.reshape(-1, pred_fair_logits.shape[-1])

            loss = criterion(pred_fair_logits, y_fair[split:])
            accuracy = (pred_fair_logits.argmax(dim=1) == y_fair[split:]).float().mean()
            datasets_loss += loss.item()
            epoch_loss += loss.item()
            datasets_accuracy += accuracy.item()
            epoch_accuracy += accuracy.item()

            wandb.log({
                "loss_step": loss.item(),
                "dataset": epoch * S + step + 1,
            })
            wandb.log({
                "accuracy_step": accuracy.item(),
                "dataset": epoch * S + step + 1,
            })

            if step % 100 == 0 and step > 0:
                wandb.log({
                    "datasets_loss": datasets_loss / 100,
                    "datasets_trained_times_100": (epoch * S + step + 1) // 100,
                })
                datasets_loss = 0
                wandb.log({
                    "datasets_accuracy": datasets_accuracy / 100,
                    "datasets_trained_times_100": (epoch * S + step + 1) // 100,
                })
                datasets_accuracy = 0

            loss.backward()
            optimizer.step()

        wandb.log({
            "epoch_loss": epoch_loss / S,
            "epoch": epoch + 1
        })
        wandb.log({
            "epoch_accuracy": epoch_accuracy / S,
            "epoch": epoch + 1
        })

        if epoch % 50 == 0 and epoch > 0:
            model.save_model(Path(f"models/fairpfn_model_epoch_{epoch + 1}.pth"))
            print(f"Model saved at epoch {epoch + 1}")
        
    return model
