import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import config
from loader import (discover_dataset_tags, select_dataset_tags, load_from_multiple,
                    load_features_results, make_scatter_loader)
from model import NetRel, NetRelH, NetRelHD
from train import scheduled_lr_train, latest_checkpoint, load_training_state


# Model name -> (class, accepts_fd). Add new architectures here to expose them on the CLI.
MODELS = {
    "NetRelHD": (NetRelHD, True),
    "NetRelH": (NetRelH, True),
    "NetRel": (NetRel, False),
}


def build_model(args, activation):
    cls, accepts_fd = MODELS[args.model]
    kwargs = dict(d=args.d, num_inputs=args.num_inputs, activation=activation)
    if accepts_fd:
        kwargs["fd"] = args.fd
    return cls(**kwargs)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a Winter evaluation network from CLI-selected datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Dataset selection
    parser.add_argument('--data-dir', type=str, default="../datasets/",
                        help="Directory holding features_desk_v*.npz / targets_desk_v*.npz")
    parser.add_argument('--datasets', type=str, nargs='+', default=None,
                        help="Datasets to train on: version tags (5, 100a), ranges (200-221), "
                             "special names (vEnd), or 'all'. Newest variant always wins "
                             "(e.g. 100 selects 100a if present).")
    parser.add_argument('--exclude', type=str, nargs='*', default=None,
                        help="Datasets to drop from the selection (same grammar as --datasets)")
    parser.add_argument('--portion', type=float, default=1.0,
                        help="Fraction of each selected dataset to use (random subsample)")
    parser.add_argument('--reload-every', type=int, default=0,
                        help="Resample the training data every N epochs instead of loading it "
                             "all once. With --portion < 1 this bounds memory (only ~portion of "
                             "the corpus is resident) while still covering it all over time. "
                             "0 (default) loads once.")
    parser.add_argument('--val-name', type=str, default="validation_games",
                        help="Name of the validation dataset (features_{name}.npz); empty to skip")
    parser.add_argument('--list', action='store_true',
                        help="List the datasets that would be loaded, then exit")

    # Model
    parser.add_argument('--model', type=str, default="NetRelHD", choices=sorted(MODELS),
                        help="Network architecture")
    parser.add_argument('--d', type=int, default=16, help="Relative-conv block width")
    parser.add_argument('--fd', type=int, default=64, help="Full hidden-layer width (H models)")
    parser.add_argument('--num-inputs', type=int, default=768, help="Number of input features used")
    parser.add_argument('--load', type=str, default=None,
                        help="Path to a checkpoint to load weights from (one-off; no schedule resume)")
    parser.add_argument('--auto-resume', action='store_true', default=False,
                        help="Resume run '--name': load newest checkpoint + LR-schedule/optimizer "
                             "state from ../models/<name>/. Safe on a fresh run (nothing to resume).")

    # Training
    parser.add_argument('--name', type=str, default="default", help="Name for model/log storage")
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--init-lr', type=float, default=0.008)
    parser.add_argument('--min-lr', type=float, default=0.0001)
    parser.add_argument('--lr-mult', type=float, default=0.5)
    parser.add_argument('--optimizer', type=str, default="sgd",
                        choices=["sgd", "ranger", "winter_ranger"],
                        help="Optimizer: sgd (momentum, historical default), ranger (Ranger2020), "
                             "or winter_ranger (RangerLite-inspired: PNM + stable weight decay + "
                             "norm loss + Lookahead)")
    parser.add_argument('--momentum', type=float, default=0.9, help="SGD momentum")
    parser.add_argument('--weight-decay', type=float, default=0.0, help="Weight decay (both optimizers)")
    parser.add_argument('--persistent-optimizer', action='store_true', default=False,
                        help="Keep one optimizer across the whole LR schedule (just change its LR "
                             "per step) instead of recreating it each step. Recommended for ranger, "
                             "which relies on long-horizon Adam/Lookahead state.")
    parser.add_argument('--eps', type=float, default=None,
                        help="Optimizer epsilon (Ranger variants). Default per optimizer: 1e-5 "
                             "ranger, 1e-7 winter_ranger. Raise it (e.g. 1e-3) to damp the "
                             "adaptive step in flat regions and curb late Adam-family divergence.")
    parser.add_argument('--beta1', type=float, default=None,
                        help="beta1 (Ranger variants). Default 0.95 ranger, 0.9 winter_ranger.")
    parser.add_argument('--beta2', type=float, default=None,
                        help="beta2 (Ranger variants). Default 0.999.")
    # winter_ranger component knobs (defaults reproduce the full RangerLite configuration)
    parser.add_argument('--normloss-factor', type=float, default=1e-4,
                        help="winter_ranger norm-loss strength (0 disables)")
    parser.add_argument('--no-pnm', action='store_true', default=False,
                        help="winter_ranger: disable Positive-Negative Momentum (use plain Adam EMA)")
    parser.add_argument('--lookahead-k', type=int, default=5,
                        help="winter_ranger Lookahead merge period in steps")
    parser.add_argument('--lookahead-alpha', type=float, default=0.5,
                        help="winter_ranger Lookahead blending factor")
    parser.add_argument('--clip-grad-norm', type=float, default=None,
                        help="Clip the total gradient norm to this value each step (safety net "
                             "against runaway gradients). Off by default.")
    parser.add_argument('--epochs-per-step', type=int, default=2)
    parser.add_argument('--log-freq', type=int, default=100000)
    parser.add_argument('--device', type=int, default=0, help="CUDA device index")
    parser.add_argument('--no-cuda', action='store_true', default=False, help="Force CPU training")

    # Logging
    parser.add_argument('--no-tensorboard', action='store_true', default=False,
                        help="Disable TensorBoard logging")
    parser.add_argument('--tb-dir', type=str, default="../logs/tb",
                        help="TensorBoard log root; this run logs to <tb-dir>/<name>")
    return parser.parse_args()


def make_writer(args):
    """Create a TensorBoard SummaryWriter for this run, or return None if disabled/unavailable."""
    if args.no_tensorboard:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        print("TensorBoard not available (pip install tensorboard); continuing without it.")
        return None
    log_dir = os.path.join(args.tb_dir, args.name)
    print(f"TensorBoard logging to {log_dir}")
    return SummaryWriter(log_dir=log_dir)


def main():
    args = parse_args()

    # load_from_multiple concatenates save_dir directly, so ensure a trailing separator.
    args.data_dir = os.path.join(args.data_dir, "")

    available = discover_dataset_tags(args.data_dir)
    if not available:
        print(f"No datasets found in {args.data_dir}")
        return 1

    if args.datasets is None:
        print("No --datasets given. Available versions:")
        print("  " + " ".join(available))
        return 1

    tags = select_dataset_tags(args.datasets, available, exclude=args.exclude)
    if not tags:
        print("Selection resolved to no datasets. Available versions:")
        print("  " + " ".join(available))
        return 1

    print(f"Selected {len(tags)} dataset(s): {' '.join(tags)}")
    if args.list:
        return 0

    config.device = torch.device("cuda:%d" % args.device if not args.no_cuda else "cpu")
    config.name = args.name
    print(f"Compute device is {config.device}")

    # A fresh scatter-loader over a (possibly subsampled) draw of the selected datasets.
    def build_train_loader():
        features, results = load_from_multiple(tags, portion=args.portion, save_dir=args.data_dir)
        print(f"Loaded {features.shape[0]} positions ({features.shape[1]} features each)")
        return make_scatter_loader(features, results, batch_size=args.batch_size,
                                   shuffle=True, device=config.device)

    # --reload-every > 0: stream fresh subsets during training (memory bounded by --portion).
    # Otherwise load the data once up front, as before.
    if args.reload_every > 0:
        train_loader = None
        data_loader_fn = build_train_loader
        print(f"Resampling: drawing a fresh {args.portion:.0%} subset every {args.reload_every} epoch(s)")
    else:
        train_loader = build_train_loader()
        data_loader_fn = None

    val_loader = None
    if args.val_name:
        vf, vr = load_features_results(args.val_name, data_dir=args.data_dir)
        val_loader = make_scatter_loader(vf, vr, batch_size=args.batch_size,
                                         shuffle=False, device=config.device)
        print(f"Validation: {vf.shape[0]} positions from {args.val_name}")

    activation = nn.Hardtanh(min_val=0, max_val=8)
    model = build_model(args, activation)

    # Resume: load newest weights + the LR-schedule/optimizer state for this run.
    # --load is a one-off weight load that does not resume the schedule.
    resume_state = None
    if args.auto_resume:
        ckpt = latest_checkpoint(args.name)
        if ckpt:
            model.load_state_dict(torch.load(ckpt, map_location="cpu"))
            print(f"Resuming: loaded weights from {ckpt}")
        resume_state = load_training_state(args.name)
        if resume_state is None:
            print(f"No prior state for run '{args.name}'; starting fresh.")
    elif args.load:
        model.load_state_dict(torch.load(args.load, map_location="cpu"))
        print(f"Loaded weights from {args.load}")
    model.to(config.device)

    # None -> let make_optimizer pick the per-optimizer default for whichever beta is unset.
    betas = None
    if args.beta1 is not None or args.beta2 is not None:
        betas = (args.beta1 if args.beta1 is not None else 0.9,
                 args.beta2 if args.beta2 is not None else 0.999)

    writer = make_writer(args)
    try:
        scheduled_lr_train(model, train_loader, val_loader=val_loader, init_lr=args.init_lr,
                           min_lr=args.min_lr, lr_mult=args.lr_mult,
                           epochs_per_step=args.epochs_per_step, log_freq=args.log_freq,
                           resume_state=resume_state, writer=writer,
                           data_loader_fn=data_loader_fn, reload_every=args.reload_every,
                           optimizer_name=args.optimizer, momentum=args.momentum,
                           weight_decay=args.weight_decay,
                           persistent_optimizer=args.persistent_optimizer,
                           eps=args.eps, betas=betas, clip_grad_norm=args.clip_grad_norm,
                           normloss_factor=args.normloss_factor, pnm=not args.no_pnm,
                           lookahead_k=args.lookahead_k, lookahead_alpha=args.lookahead_alpha)
    finally:
        if writer is not None:
            writer.close()
    return 0


if __name__ == '__main__':
    print("torch version", torch.__version__)
    raise SystemExit(main())
