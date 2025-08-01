import yaml
import wandb
from pathlib import Path
from train import pretrain_fairpfn, pretrain_fairpfn_on_pre_generated_data

CONFIG_FILE = "configs.yml"

def main():
    with open(CONFIG_FILE, "r") as file:
        config = yaml.safe_load(file)

    wandb.init(
        project="FairPFN",
        name=config["wandb_name"],
        config=config
    )

    if config["use_pre_genereated_data"]:
        pretrain_fairpfn_on_pre_generated_data(
            E=config["E"],
            batch_size=config["batch_size"],
            filename=Path(config["pregen_data_path"]),
            lr = float(config["learning_rate"]),
        )
    else:
        pretrain_fairpfn(
            E=config["E"],
            S=config["S"],
            U=config["U"],
            H=config["H"],
            M=config["M"],
            N=config["N"],
            lr = float(config["learning_rate"]),
        )

if __name__ == "__main__":
    main()

