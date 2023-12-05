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
    parser.add_argument('--name', type=str, default="default", help="Name for model and log storage")
    parser.add_argument('--init-lr', type=float, default=0.008, help="Initial training learning rate")
    parser.add_argument('--min-lr', type=float, default=0.0001, help="Minimum training learning rate")
    parser.add_argument('--epochs-per-step', type=int, default=2)
    parser.add_argument('--log-freq', type=int, default=100000)
    args = parser.parse_args()

    f, r = load_from_multiple([2, 5, 6, 7, 8, 9, 10, 11, 12], save_dir="../datasets/")
    val_loader = load_dataset_ocb("../datasets/validation_games", batch_size=25, shuffle=False)
    data_loader = torch.utils.data.DataLoader(CSRDataset(f, r), batch_size=args.batch_size, shuffle=True)

    config.device = torch.device("cuda:%d" % args.device if not args.no_cuda else "cpu")
    print(f"Compute device is {config.device}")

    config.name = args.name

    model = NetRel(d=8, num_inputs=773, activation=nn.Hardtanh(min_val=0, max_val=8))

    model.to(config.device)
    scheduled_lr_train(model, data_loader, val_loader=val_loader, init_lr=args.init_lr, min_lr=args.min_lr,
                       epochs_per_step=args.epochs_per_step, log_freq=args.log_freq)
    return 0


if __name__ == '__main__':
    print("torch version", torch.__version__)
    main()
