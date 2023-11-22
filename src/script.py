import argparse
import torch
import torch.nn as nn

from loader import load_dataset_ocb, CSRDataset, load_from_multiple
from model import NetRel
from train import scheduled_lr_train
import config


def main():
    # Command line argument parsing
    parser = argparse.ArgumentParser(description="Training Script for Teacher Models")
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()

    f, r = load_from_multiple([2, 5, 6, 7, 8, 9, 10, 11, 12], save_dir="../datasets/")
    val_loader = load_dataset_ocb("../datasets/validation_games", batch_size=25, shuffle=False)
    data_loader = torch.utils.data.DataLoader(CSRDataset(f, r), batch_size=args.batch_size, shuffle=True)

    config.device = torch.device("cuda:%d" % args.device if not args.no_cuda else "cpu")
    print(f"Compute device is {config.device}")

    model = NetRel(d=8, num_inputs=773, activation=nn.Hardtanh(min_val=0, max_val=8))

    model.to(config.device)
    scheduled_lr_train(model, "rn8S01a0", data_loader, val_loader=val_loader, init_lr=0.008, min_lr=0.008,
                       epochs_per_step=2, log_freq=10000)
    return 0


if __name__ == '__main__':
    print("torch version", torch.__version__)
    main()
