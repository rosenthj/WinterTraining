import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import config
from loader import (discover_dataset_tags, select_dataset_tags, load_from_multiple,
                    load_features_results, make_scatter_loader)
from model import NetRel, NetRelH, NetRelHD
from train import (scheduled_lr_train, latest_checkpoint, load_training_state,
                   load_partial_state_dict)


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
    parser.add_argument('--aux-data-dir', type=str, default="../datasets_aux/",
                        help="Directory holding auxiliary datasets that must be selected "
                             "individually (never included by --datasets all)")
    parser.add_argument('--aux-datasets', type=str, nargs='+', default=None,
                        help="Auxiliary datasets to also train on, resolved against "
                             "--aux-data-dir. Same grammar as --datasets except 'all' is "
                             "not allowed: auxiliary datasets are opt-in by name.")
    parser.add_argument('--portion', type=float, default=1.0,
                        help="Fraction of each selected dataset to use (random subsample)")
    parser.add_argument('--aux-portion', type=float, default=None,
                        help="Portion applied to --aux-datasets instead of --portion. "
                             "Values > 1 oversample (replicate) the aux data, raising its "
                             "share of each batch and its per-epoch gradient mass. Defaults "
                             "to --portion when unset.")
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
    parser.add_argument('--init-from', type=str, default=None,
                        help="Fine-tune/retrain: initialize a FRESH run's weights from this "
                             "checkpoint, then train a new schedule. Superseded by --auto-resume "
                             "once the run has its own checkpoints, so it composes with "
                             "submit_chain.sh (segment 1 seeds from the base, rest resume). The "
                             "architecture flags must match the checkpoint.")
    parser.add_argument('--init-partial', action='store_true', default=False,
                        help="Grow the net: allow --init-from (or --load) to seed a LARGER "
                             "model from a smaller checkpoint. Copies the overlapping slice of "
                             "each shared parameter; newly added units keep their random init "
                             "and mismatched/missing layers are left untouched. Use when "
                             "bumping --d / --fd above the checkpoint's size.")
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
    parser.add_argument('--reg-weights-only', action='store_true', default=False,
                        help="Apply weight decay and norm loss only to Linear/Conv weights, "
                             "exempting biases and bias-like params (e.g. b1). Recommended for "
                             "this architecture; mainly affects winter_ranger's norm loss.")
    parser.add_argument('--clip-grad-norm', type=float, default=None,
                        help="Clip the total gradient norm to this value each step (safety net "
                             "against runaway gradients). Off by default.")
    parser.add_argument('--ce-weight', type=float, default=0.04,
                        help="Weight of the cross-entropy term in the optimized loss "
                             "(total = reg + ce_weight*ce + draw_weight*draw). Default 0.04 "
                             "(historical value). CE is a log-barrier term: it guards the tails "
                             "(punishes near-zero probability on the realized outcome).")
    parser.add_argument('--draw-weight', type=float, default=0.0,
                        help="Weight of the draw-MSE term (MSE of the draw probability vs the "
                             "draw indicator). Default 0 (off). Covers the draw axis the "
                             "regression term is blind to, with a bounded quadratic penalty; "
                             "reg + 3*draw is the multiclass Brier score, so 3 reproduces Brier "
                             "while smaller values keep the eval axis dominant. Set --ce-weight 0 "
                             "to use draw-MSE instead of CE, or both for a combined loss.")
    parser.add_argument('--epochs-per-step', type=int, default=2,
                        help="step schedule: epochs between each LR *lr_mult drop (ignored for wsd)")
    parser.add_argument('--schedule', type=str, default="step", choices=["step", "wsd"],
                        help="LR schedule. 'step' (default): geometric decay init_lr*lr_mult^"
                             "(epoch//epochs_per_step) until < min_lr (historical; kept for "
                             "reproducing older runs). 'wsd': Warmup-Stable-Decay over a fixed "
                             "--total-epochs budget -- linear warmup (--warmup-steps) to peak "
                             "--init-lr, a long stable phase at the peak, then a half-cosine decay "
                             "to --min-lr over the final --decay-frac of the run. Uses one "
                             "persistent optimizer; --lr-mult/--epochs-per-step are ignored.")
    parser.add_argument('--total-epochs', type=int, default=None,
                        help="wsd: total training budget in epochs (required for --schedule wsd). "
                             "Sizes the stable/decay windows; the run stops after this many epochs.")
    parser.add_argument('--warmup-steps', type=int, default=0,
                        help="wsd: linear LR warmup length in batches (from ~0 to peak --init-lr). "
                             "Counts from this run's global_step 0, so a grown/fresh run re-warms "
                             "at its start while a resumed segment does not. Try a few hundred to a "
                             "few thousand; especially useful right after growing the net.")
    parser.add_argument('--decay-frac', type=float, default=0.1,
                        help="wsd: fraction of the total run spent in the final cosine decay to "
                             "--min-lr (e.g. 0.1 = last 10%%). The rest (after warmup) is stable.")
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

    if args.schedule == "wsd" and not (args.total_epochs and args.total_epochs > 0):
        print("Error: --schedule wsd requires --total-epochs > 0 (the training budget).")
        return 1

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

    aux_tags = []
    if args.aux_datasets:
        if any(token.strip().lower() == "all" for token in args.aux_datasets):
            print("Error: 'all' is not allowed for --aux-datasets; "
                  "auxiliary datasets must be named individually.")
            return 1
        args.aux_data_dir = os.path.join(args.aux_data_dir, "")
        aux_available = discover_dataset_tags(args.aux_data_dir)
        if not aux_available:
            print(f"No datasets found in {args.aux_data_dir}")
            return 1
        aux_tags = select_dataset_tags(args.aux_datasets, aux_available)
        if not aux_tags:
            print(f"--aux-datasets resolved to no datasets. Available in {args.aux_data_dir}:")
            print("  " + " ".join(aux_available))
            return 1
        print(f"Selected {len(aux_tags)} auxiliary dataset(s): {' '.join(aux_tags)}")

    if args.list:
        return 0

    config.device = torch.device("cuda:%d" % args.device if not args.no_cuda else "cpu")
    config.name = args.name
    print(f"Compute device is {config.device}")

    # A fresh scatter-loader over a (possibly subsampled) draw of the selected datasets.
    def build_train_loader():
        aux_portion = args.portion if args.aux_portion is None else args.aux_portion
        selection = list(tags) + [(tag, aux_portion, args.aux_data_dir) for tag in aux_tags]
        features, results = load_from_multiple(selection, portion=args.portion,
                                               save_dir=args.data_dir)
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

    # Decide the model's starting weights. Precedence under --auto-resume: an existing
    # checkpoint for this run (continue it) > --init-from (fresh fine-tune) > random init.
    # --init-from is thus superseded once a run has its own checkpoints, so it composes with
    # the segment chain: segment 1 seeds from the base model, later segments resume normally.
    # --load is a plain one-off weight load (no schedule resume) for non-resume use.
    def seed_from(path):
        # Partial (grow-the-net) or strict load, per --init-partial.
        state = torch.load(path, map_location="cpu")
        if args.init_partial:
            load_partial_state_dict(model, state)
        else:
            model.load_state_dict(state)

    resume_state = None
    if args.auto_resume:
        ckpt = latest_checkpoint(args.name)
        if ckpt:
            # A resumed run continues at its own size; never partial-load a checkpoint.
            model.load_state_dict(torch.load(ckpt, map_location="cpu"))
            resume_state = load_training_state(args.name)
            print(f"Resuming run '{args.name}': loaded weights from {ckpt}")
        elif args.init_from:
            seed_from(args.init_from)
            print(f"Fine-tuning: initialized run '{args.name}' from {args.init_from}")
        else:
            print(f"Starting run '{args.name}' fresh.")
    elif args.init_from:
        seed_from(args.init_from)
        print(f"Fine-tuning: initialized from {args.init_from}")
    elif args.load:
        seed_from(args.load)
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
                           lookahead_k=args.lookahead_k, lookahead_alpha=args.lookahead_alpha,
                           reg_weights_only=args.reg_weights_only, ce_weight=args.ce_weight,
                           draw_weight=args.draw_weight, schedule=args.schedule,
                           total_epochs=args.total_epochs, warmup_steps=args.warmup_steps,
                           decay_frac=args.decay_frac)
    finally:
        if writer is not None:
            writer.close()
    return 0


if __name__ == '__main__':
    print("torch version", torch.__version__)
    raise SystemExit(main())
