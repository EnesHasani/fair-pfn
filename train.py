import torch
import torch.nn as nn
import torch.optim as optim
from model import FairPFNModel
from data_generator import DataGenerator


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
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(E):
        print(f"Epoch {epoch+1}/{E}")
        for step in range(S):
            generator = DataGenerator(U=U, H=H, M=M, N=N, device=device)
            Dbias, y_fair = generator.generate_dataset()


            split = int(0.7 * len(Dbias))

            train_biased_features = Dbias[:split, :-1].unsqueeze(1).to(device)  
            train_biased_labels = Dbias[:split, -1].unsqueeze(1).to(device)
            val_biased_features = Dbias[split:, :-1].unsqueeze(1).to(device)
            num_classes = len(torch.unique(train_biased_labels))

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

            loss.backward()
            optimizer.step()
            if step % 100 == 0:
                print(f"  Step {step}/{S} - Loss: {loss.item():.4f}")

    return model
