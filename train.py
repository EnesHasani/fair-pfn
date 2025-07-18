import torch
import torch.nn as nn
import torch.optim as optim
from model import FairPFNModel
from datasets import FairPFNDataset
from mlp_generator import MLPGenerator


def pretrain_fairpfn(
    E: int,               # Epochs
    S: int,               # Steps per epoch
    U: int = 8,           # Exogenous variables
    H: int = 3,           # MLP depth
    M: int = 6,           # Features
    N: int = 128,         # Samples per dataset
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    model = FairPFNModel(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(E):
        print(f"Epoch {epoch+1}/{E}")
        for step in range(S):
            generator = MLPGenerator(U=U, H=H, M=M, N=N)
            Dbias, Dfair = generator.generate_dataset()

            split = int(0.7 * len(Dbias))           #question: How should I split the dataset?
            train_data = FairPFNDataset(Dbias[:split], Dfair[:split])
            val_data = FairPFNDataset(Dbias[split:], Dfair[split:])


            train_biased_features = torch.cat((train_data.A, train_data.X), dim=1).unsqueeze(1).to(device)
            train_biased_labels = train_data.y_biased.unsqueeze(1).to(device)
            val_biased_features = torch.cat((val_data.A, val_data.X), dim=1).unsqueeze(1).to(device)
            val_fair_labels = val_data.y_fair.to(device)
            num_classes = len(torch.unique(train_biased_labels))

            # Shapes
            print(f"  Train biased features shape: {train_biased_features.shape}")
            print(f"  Train biased labels shape: {train_biased_labels.shape}")
            print(f"  Val biased features shape: {val_biased_features.shape}")
            print(f"  Val fair labels shape: {val_fair_labels.shape}")

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
            loss = criterion(pred_fair_logits, val_fair_labels.squeeze(1).long())

            loss.backward()
            optimizer.step()
            if step % 100 == 0:
                print(f"  Step {step}/{S} - Loss: {loss.item():.4f}")

    return model
