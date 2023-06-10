import torch
import torch.nn as nn

from loader import load_dataset_ocb, CSRDataset, load_from_multiple
from model import NetRel
from train import scheduled_lr_train


def main():
    f, r = load_from_multiple([2, 5, 6, 7, 8, 9, 10, 11, 12], save_dir="../datasets/")
    dloader = torch.utils.data.DataLoader(CSRDataset(f, r), batch_size=16, shuffle=True)
    val_loader = load_dataset_ocb("validation_games", batch_size=25, shuffle=False)

    model = NetRel(d=8, num_inputs=773, activation=nn.Hardtanh(min_val=0, max_val=8))
    scheduled_lr_train(model, "rn8S01a0", dloader, val_loader=val_loader, init_lr=0.008, min_lr=0.008,
                       epochs_per_step=2, log_freq=1000)
    return 0


if __name__ == '__main__':
    print("torch version", torch.__version__)
    main()
