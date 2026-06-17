import config
import glob
import numpy as np
import os
import time
import torch
import torch.nn.functional as F

from loader import load_from_multiple, make_scatter_loader
from utils import log


def loss_f(pred, target, base_loss=F.mse_loss):
    return base_loss(pred[:, 0] - pred[:, 2], 1.0 - target)


def loss_components(pred_logit, target, base_loss=F.mse_loss):
    """Return ``(total, reg, ce)``: the optimized loss and its two components.

    ``reg`` is the WDL regression term (``base_loss`` on the win-minus-loss probability),
    ``ce`` is the cross-entropy term, and ``total = reg + 0.04 * ce`` is what is optimized.
    """
    pred = F.softmax(pred_logit, dim=-1)
    reg = loss_f(pred, target, base_loss)
    ce = F.cross_entropy(pred_logit, target)
    return reg + 0.04 * ce, reg, ce


def loss_combined(pred_logit, target, base_loss=F.mse_loss):
    return loss_components(pred_logit, target, base_loss)[0]


def randomize_piece_positions(x, randomization):
    if isinstance(randomization, bool):
        if not randomization:
            return x
        randomization = np.arange(6)
    if isinstance(randomization, str):
        raise NotImplemented("Custom randomization not yet implemented")
    x = x.clone()
    for i in range(12):
        pt = i % 6
        if pt not in randomization:
            continue
        lb = i * 64
        ub = (i + 1) * 64
        x[:, lb:ub] = x[:, lb:ub][:, torch.randperm(64)]
    return x


def test(model, test_loader, base_loss=F.mse_loss, rec=None):
    if not isinstance(base_loss, list):
        base_loss = [base_loss]
    training = model.training
    model.eval()
    loss_sum = np.zeros(len(base_loss))
    count = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(config.device), target.to(config.device)
            if rec is None:
                output = model(data.type(torch.float32).to(config.device))
            else:
                output = model(data.type(torch.float32).to(config.device), rec=rec)
            for i in range(len(base_loss)):
                loss = loss_f(output, target, base_loss=base_loss[i])
                loss_sum[i] += loss.item()
            count += 1
    model.train(training)
    return loss_sum / count


def get_h_mirrored_position_tensor(board_tensor):
    assert len(board_tensor.shape) == 1 and board_tensor.shape[0] == 772
    res = torch.zeros(772)
    for piecetype in range(12):
        for square in range(64):
            x = square % 8
            y = square // 8
            h_square = y * 8 + 7 - x
            res[piecetype * 64 + square] = board_tensor[piecetype * 64 + h_square]
    return res


def gen_mirror_dataset(data_loader, count):
    positions = []
    mirrored = []
    for (features, targets) in data_loader:
        for i in range(features.shape[0]):
            if features[i, 768:].sum() == 0:
                positions.append(features[i])
                mirrored.append(get_h_mirrored_position_tensor(features[i]))
                if len(positions) >= count:
                    return torch.stack(positions), torch.stack(mirrored)
    return torch.stack(positions), torch.stack(mirrored)


def gen_validation_string(model, validation_loader, rec=None):
    test_loss = test(model, validation_loader, base_loss=[F.mse_loss, F.l1_loss], rec=rec)
    # start_pred, start_eval = get_startpos_eval(model)
    # wdl_str = f"({start_pred[0]:.4f}/{start_pred[1]:.4f}/{start_pred[2]:.4f})"
    # return f"Val mse:{test_loss[0]:.6f}, Val l1:{test_loss[1]:.6f}, Start WDL:{wdl_str}=>{start_eval:.5f}"
    return f"Val mse:{test_loss[0]:.6f}, Val l1:{test_loss[1]:.6f}"


def _grad_norm(model):
    """Total L2 norm of the current gradients (a snapshot of the most recent batch)."""
    sq = None
    for p in model.parameters():
        if p.grad is not None:
            s = p.grad.detach().pow(2).sum()
            sq = s if sq is None else sq + s
    return sq.sqrt().item() if sq is not None else None


class _ActSaturationMeter:
    """Forward hook tallying how much of the clipped-ReLU activation is pinned at the
    bottom (0) or top (max_val) of its range -- i.e. zero-gradient ("dead") activations.

    Buckets by tensor rank so the convolutional feature map (4D) and the fully-connected
    hidden layer (2D) are reported separately. This matters for the relative net: the conv
    activation is multiplied by the piece-presence mask, so its bottom-clamp count is
    dominated by structural zeros (empty squares), not dead units. The fc bucket and both
    ``*_frac_max`` signals are therefore the clean indicators of genuinely dead neurons.
    """

    def __init__(self, max_val, eps=1e-4):
        self.max_val = max_val
        self.eps = eps
        self.stats = {"conv": [0, 0, 0], "fc": [0, 0, 0]}  # [num_zero, num_max, total]

    def __call__(self, module, inputs, output):
        bucket = "fc" if output.dim() <= 2 else "conv"
        s = self.stats[bucket]
        s[0] += int((output <= self.eps).sum().item())
        s[1] += int((output >= self.max_val - self.eps).sum().item())
        s[2] += output.numel()


def log_activation_saturation(model, data, writer, global_step):
    """Log the fraction of saturated (dead) clipped-ReLU activations on one batch.

    Uses a hooked no-grad forward with RNG and train/eval state saved and restored, so it
    has no effect on training. Called only at log points, so the extra forward is cheap.
    """
    activation = getattr(model, "activation", None)
    if not isinstance(activation, torch.nn.Module):
        return
    max_val = getattr(activation, "max_val", 8.0)
    meter = _ActSaturationMeter(max_val)
    handle = activation.register_forward_hook(meter)
    rng_state = torch.get_rng_state()
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            if config.rec is None:
                model(data.type(torch.float32), activate=False)
            else:
                model(data.type(torch.float32), activate=False, rec=0)
    finally:
        handle.remove()
        model.train(was_training)
        torch.set_rng_state(rng_state)
    for bucket, (num_zero, num_max, total) in meter.stats.items():
        if total:
            writer.add_scalar(f"act/{bucket}_frac_zero", num_zero / total, global_step)
            writer.add_scalar(f"act/{bucket}_frac_max", num_max / total, global_step)


def train_epoch(model, optimizer, train_loader, log_freq=1000, rng_piece_positions=False, base_loss=F.mse_loss,
                test_loader=None, name=None, writer=None, global_step=0, lr=None, clip_grad_norm=None):
    model.train()
    # Running sums of [total, reg, ce] kept on-device so we only synchronise (.item()) once
    # per log_freq batches instead of every batch -- cheaper than the previous per-batch
    # loss.item() while still exposing the loss components.
    epoch_sums = torch.zeros(3, device=config.device)
    recent_sums = torch.zeros(3, device=config.device)
    count = 0
    recent_count = 0
    recent_positions = 0
    recent_t0 = time.perf_counter()
    # Pre-clip gradient norm of the most recent batch (returned by clip_grad_norm_), so the
    # train/grad_norm metric still shows true spikes even when clipping is active.
    last_grad_norm = None

    def flush(include_val):
        # Emit one set of scalars for the batches since the last flush. Called every
        # log_freq batches and once more at the end of the epoch (so every epoch yields a
        # point even when an epoch is shorter than log_freq, and so train/lr has a sample
        # right before each step change -- which makes it render as a clean step function
        # rather than a linearly-interpolated ramp).
        nonlocal recent_sums, recent_count, recent_positions, recent_t0
        if recent_count == 0:
            return
        # Measure the training time for this window before running validation, so the
        # validation pass does not depress the throughput metric.
        dt = time.perf_counter() - recent_t0
        recent = (recent_sums / recent_count).tolist()  # single host sync
        total_avg = (epoch_sums[0] / count).item()
        msg = (f"Batch {count} Recent:{recent[0]:.6f} (reg {recent[1]:.6f}, ce {recent[2]:.6f}), "
               f"Total:{total_avg:.6f}")
        val_losses = None
        if include_val and test_loader is not None:
            if config.rec is not None and config.rec > 1:
                msg += ", " + gen_validation_string(model, test_loader, rec=1)
            # Compute the validation losses once and reuse them for both the log line and
            # the TensorBoard scalars (this is the same test() the log already ran).
            val_losses = test(model, test_loader, base_loss=[F.mse_loss, F.l1_loss])
            msg += f", Val mse:{val_losses[0]:.6f}, Val l1:{val_losses[1]:.6f}"
        log(msg)
        if writer is not None:
            writer.add_scalar("train/loss", recent[0], global_step)
            writer.add_scalar("train/loss_reg", recent[1], global_step)
            writer.add_scalar("train/loss_ce", recent[2], global_step)
            if lr is not None:
                writer.add_scalar("train/lr", lr, global_step)
            # Prefer the pre-clip norm captured during the step; fall back to a direct
            # measurement when clipping is off.
            grad_norm = last_grad_norm.item() if last_grad_norm is not None else _grad_norm(model)
            if grad_norm is not None:
                writer.add_scalar("train/grad_norm", grad_norm, global_step)
            if dt > 0:
                writer.add_scalar("train/positions_per_sec", recent_positions / dt, global_step)
            if val_losses is not None:
                writer.add_scalar("val/mse", val_losses[0], global_step)
                writer.add_scalar("val/l1", val_losses[1], global_step)
            log_activation_saturation(model, data, writer, global_step)
        if name is not None:
            save(model, name=name)
        recent_sums = torch.zeros(3, device=config.device)
        recent_count = 0
        recent_positions = 0
        recent_t0 = time.perf_counter()

    for (data, target) in train_loader:
        # data = randomize_piece_positions(data, rng_piece_positions)
        data, target = data.to(config.device), target.to(config.device)
        optimizer.zero_grad()

        if config.rec is None:
            output = model(data.type(torch.float32), activate=False)
        else:
            output = model(data.type(torch.float32), activate=False, rec=torch.randint(config.rec, (1,)).item())
        total, reg, ce = loss_components(output, target, base_loss=base_loss)
        total.backward()
        if clip_grad_norm:
            # clip_grad_norm_ returns the total norm *before* clipping (kept on-device; only
            # synced to host at log time) and rescales the gradients in place.
            last_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        optimizer.step()

        with torch.no_grad():
            batch_stats = torch.stack([total.detach(), reg.detach(), ce.detach()])
            epoch_sums += batch_stats
            recent_sums += batch_stats
        count += 1
        recent_count += 1
        recent_positions += target.shape[0]
        global_step += 1
        if count % log_freq == 0:
            flush(include_val=True)
    flush(include_val=False)  # capture the tail of the epoch (no extra validation pass)
    avg_loss = (epoch_sums[0] / count).item() if count else 0.0
    return avg_loss, global_step


def save(model, path=None, name=None, epoch=None):
    if path is None:
        assert name is not None
        if epoch is None:
            path = f"../models/{name}/{name}_tmp"
        else:
            path = f"../models/{name}/{name}_ep{epoch + 1}"
    torch.save(model.state_dict(), f"{path}.pt")
    model.serialize(f"{path}.bin", verbose=1)


def train(model, train_loader, epochs, optimizer=None, lr=0.01, log_freq=100000, loss=F.mse_loss, initial_epoch=0,
          test_loader=None):
    if optimizer is None:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    if not os.path.exists(f"../models/{config.name}"):
        os.makedirs(f"../models/{config.name}")
    for epoch in range(initial_epoch, initial_epoch + epochs):
        if epoch != initial_epoch:
            print()
        log(
            f"Epoch {epoch + 1}--Training on {len(train_loader.dataset)} samples----------------------------------------------------")
        train_epoch(model, optimizer, train_loader, log_freq=log_freq, base_loss=loss, test_loader=test_loader,
                    name=config.name)
        save(model, f"../models/{config.name}/{config.name}_ep{epoch + 1}")
        # torch.save(model.state_dict(), f"../models/{name}/{name}_ep{epoch + 1}.pt")
        # model.serialize(f"../models/{name}/{name}_ep{epoch + 1}.bin", verbose=1)
        if test_loader is not None:
            log(f"Finished Epoch {epoch + 1}. {gen_validation_string(model, test_loader)}")


class OutputHook(list):
    """ Hook to capture module outputs.
    """
    def __call__(self, module, input, output):
        self.append(output)


def training_state_path(name):
    return f"../models/{name}/{name}.state.pt"


def save_training_state(name, optimizer, next_epoch, step, global_step=0):
    """Persist the schedule position and optimizer state so a later job can resume."""
    torch.save({"next_epoch": next_epoch, "step": step, "global_step": global_step,
                "optimizer": optimizer.state_dict()},
               training_state_path(name))


def load_training_state(name):
    """Return the saved schedule/optimizer state for ``name``, or None if absent."""
    path = training_state_path(name)
    if os.path.exists(path):
        return torch.load(path, map_location="cpu")
    return None


def latest_checkpoint(name):
    """Return the most recently modified weight checkpoint (.pt) for ``name``, or None.

    Matches ``{name}_*.pt`` (e.g. ``{name}_ep3.pt``, ``{name}_tmp.pt``) but not the
    ``{name}.state.pt`` schedule file.
    """
    paths = glob.glob(f"../models/{name}/{name}_*.pt")
    return max(paths, key=os.path.getmtime) if paths else None


def make_optimizer(name, params, lr, momentum=0.9, weight_decay=0.0, eps=None, betas=None,
                   normloss_factor=1e-4, pnm=True, lookahead_k=5, lookahead_alpha=0.5):
    """Build the optimizer selected on the CLI. SGD+momentum is the historical default.

    ``eps`` and ``betas`` default to None, meaning "use the optimizer's own default"
    (so each variant gets its appropriate scale). They apply to the Ranger variants only.
    Options:
      - sgd: SGD with momentum (best performer historically).
      - ranger: Ranger2020 (RAdam + Lookahead + gradient centralization, ranger.py).
      - winter_ranger: RangerLite-inspired (PNM + stable weight decay + norm loss +
        Lookahead, winter_ranger.py); the extra args tune its components.
    """
    name = name.lower()
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    if name == "ranger":
        from ranger import Ranger
        return Ranger(params, lr=lr, weight_decay=weight_decay,
                      eps=1e-5 if eps is None else eps,
                      betas=(0.95, 0.999) if betas is None else betas)
    if name == "winter_ranger":
        from winter_ranger import WinterRanger
        return WinterRanger(params, lr=lr, weight_decay=weight_decay,
                            eps=1e-7 if eps is None else eps,
                            betas=(0.9, 0.999) if betas is None else betas,
                            pnm=pnm, normloss_factor=normloss_factor,
                            lookahead_k=lookahead_k, lookahead_alpha=lookahead_alpha)
    raise ValueError(f"Unknown optimizer '{name}' (expected 'sgd', 'ranger' or 'winter_ranger')")


def _set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group["lr"] = lr


def split_regularized_params(model):
    """Split params into (regularized weight matrices, everything else).

    Only the ``.weight`` of Linear/Conv layers is regularized (weight decay + norm loss);
    biases and standalone bias-like parameters (e.g. NetRelHD.b1, a per-(channel,square)
    bias map) are excluded, since norm loss and weight decay assume a weight matrix whose
    first dimension indexes neurons -- applying them to a bias is not meaningful.
    """
    weight_ids = set()
    regularized = []
    for module in model.modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.modules.conv._ConvNd)):
            w = getattr(module, "weight", None)
            if w is not None:
                regularized.append(w)
                weight_ids.add(id(w))
    others = [p for p in model.parameters() if id(p) not in weight_ids]
    return regularized, others


def _optimizer_params(model, reg_weights_only):
    """Optimizer ``params`` argument: a single group, or two groups that exempt biases /
    bias-like parameters from weight decay and norm loss when ``reg_weights_only``."""
    if not reg_weights_only:
        return model.parameters()
    regularized, others = split_regularized_params(model)
    return [
        {"params": regularized},
        {"params": others, "weight_decay": 0.0, "normloss_factor": 0.0},
    ]


def _move_optimizer_state(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def scheduled_lr_train(model, data_loader=None, val_loader=None, loss=F.mse_loss, init_lr=0.001, min_lr=0.0001,
                       lr_mult=0.5, epochs_per_step=1, log_freq=100000, resume_state=None, writer=None,
                       data_loader_fn=None, reload_every=0, optimizer_name="sgd", momentum=0.9,
                       weight_decay=0.0, persistent_optimizer=False, eps=None, betas=None,
                       clip_grad_norm=None, normloss_factor=1e-4, pnm=True, lookahead_k=5,
                       lookahead_alpha=0.5, reg_weights_only=False):
    """Train with a decaying LR schedule.

    Either pass a fixed ``data_loader``, or pass ``data_loader_fn`` (a callable returning a
    fresh loader) together with ``reload_every`` > 0 to resample the training data every
    ``reload_every`` epochs. Resampling keeps only a subset of the corpus resident at once
    (see ``loader.load_from_multiple``'s ``portion``) while still covering all of it over
    time -- this is the memory-bounded streaming used for the full dataset.
    """
    assert 1 > lr_mult > 0, f"Unexpected lr_mult param:{lr_mult}"
    assert data_loader is not None or data_loader_fn is not None, "Provide data_loader or data_loader_fn"
    # config.activation_hook = OutputHook()
    # model.activation.register_forward_hook(config.activation_hook)
    if not os.path.exists(f"../models/{config.name}"):
        os.makedirs(f"../models/{config.name}")

    # The schedule is driven by a global epoch counter: step = epoch // epochs_per_step,
    # lr = init_lr * lr_mult**step. This makes the position fully derivable from one
    # integer, so a restarted job can pick up the schedule exactly where it stopped.
    # global_step (total batches seen) is persisted too so TensorBoard's x-axis stays
    # continuous across restarts.
    start_epoch = 0
    global_step = 0
    pending_opt_state = None
    pending_opt_step = None
    if resume_state is not None:
        start_epoch = resume_state.get("next_epoch", 0)
        global_step = resume_state.get("global_step", 0)
        pending_opt_state = resume_state.get("optimizer")
        pending_opt_step = resume_state.get("step")
        log(f"Resuming schedule at epoch {start_epoch + 1} (step {start_epoch // epochs_per_step})")

    # The start-position eval is a cheap, interpretable sanity metric: the network's
    # predicted expected score / WDL for the opening position, which should settle near a
    # small white advantage as training progresses. Imported lazily (needs python-chess);
    # if unavailable it is simply skipped.
    startpos_eval = None
    if writer is not None:
        try:
            from chess_utils import get_startpos_eval as startpos_eval
        except Exception as e:
            log(f"start-position metric unavailable: {e}")

    epoch = start_epoch
    optimizer = None
    cur_step = None
    while True:
        step = epoch // epochs_per_step
        lr = init_lr * (lr_mult ** step)
        if lr < min_lr:
            break
        # Optimizer handling at each LR-schedule step:
        #  - default: recreate a fresh optimizer per step (the original behaviour);
        #  - persistent_optimizer: build it once and span the whole run, just updating the
        #    learning rate at each step -- this preserves Adam/Lookahead state across LR
        #    drops, which is how Ranger is meant to be run.
        if optimizer is None:
            optimizer = make_optimizer(optimizer_name, _optimizer_params(model, reg_weights_only), lr,
                                       momentum=momentum, weight_decay=weight_decay, eps=eps, betas=betas,
                                       normloss_factor=normloss_factor, pnm=pnm,
                                       lookahead_k=lookahead_k, lookahead_alpha=lookahead_alpha)
            # On resume restore the saved state: always in persistent mode (one continuous
            # optimizer), otherwise only when it belongs to this step (a fresh optimizer is
            # wanted when resuming exactly at a new step boundary).
            if pending_opt_state is not None and (persistent_optimizer or pending_opt_step == step):
                optimizer.load_state_dict(pending_opt_state)
                _move_optimizer_state(optimizer, config.device)
                _set_lr(optimizer, lr)  # the restored state may carry an older step's lr
                log(f"Restored optimizer state ({'persistent' if persistent_optimizer else f'step {step}'})")
            pending_opt_state = None
            cur_step = step
            log(f"\nLearning rate is {lr:g} (step {step})")
        elif step != cur_step:
            if persistent_optimizer:
                _set_lr(optimizer, lr)
            else:
                optimizer = make_optimizer(optimizer_name, model.parameters(), lr, momentum=momentum,
                                           weight_decay=weight_decay)
            cur_step = step
            log(f"\nLearning rate is {lr:g} (step {step})")
        # In resampling mode, draw a fresh random subset every reload_every epochs (and on
        # the first epoch of a (re)started run, where data_loader has not been built yet).
        if data_loader_fn is not None and (data_loader is None or epoch % reload_every == 0):
            data_loader = data_loader_fn()
        log(f"Epoch {epoch + 1}--Training on {len(data_loader.dataset)} samples"
            "----------------------------------------------------")
        # lr is logged densely inside train_epoch (at each log point + epoch end) so the
        # train/lr curve reads as a step function instead of an interpolated ramp.
        _, global_step = train_epoch(model, optimizer, data_loader, log_freq=log_freq, base_loss=loss,
                                     test_loader=val_loader, name=config.name, writer=writer,
                                     global_step=global_step, lr=lr, clip_grad_norm=clip_grad_norm)
        epoch += 1
        save(model, f"../models/{config.name}/{config.name}_ep{epoch}")
        save_training_state(config.name, optimizer, next_epoch=epoch, step=step, global_step=global_step)
        if val_loader is not None:
            val_losses = test(model, val_loader, base_loss=[F.mse_loss, F.l1_loss])
            log(f"Finished Epoch {epoch}. Val mse:{val_losses[0]:.6f}, Val l1:{val_losses[1]:.6f}")
            if writer is not None:
                writer.add_scalar("val/mse", val_losses[0], global_step)
                writer.add_scalar("val/l1", val_losses[1], global_step)
        if writer is not None and startpos_eval is not None:
            try:
                pred, score = startpos_eval(model)
                pred = pred.tolist()
                writer.add_scalar("startpos/score", float(score), global_step)
                writer.add_scalar("startpos/win", pred[0], global_step)
                writer.add_scalar("startpos/draw", pred[1], global_step)
                writer.add_scalar("startpos/loss", pred[2], global_step)
            except Exception as e:
                log(f"start-position eval failed, disabling: {e}")
                startpos_eval = None


def train_v2(model, data_lst, portion, iters, val_loader=None, loss=F.mse_loss, init_lr=0.001, min_lr=0.0001, lr_mult=0.5,
                       epochs_per_step=1, log_freq=100000, batch_size=16):
    assert 1 > lr_mult > 0, f"Unexpected lr_mult param:{lr_mult}"
    step = 0
    lr = init_lr
    while lr >= min_lr:
        for iter in range(iters):
            f, r = load_from_multiple(data_lst, portion=portion, save_dir="../datasets/")
            data_loader = make_scatter_loader(f, r, batch_size=batch_size, shuffle=True, device=config.device)
            train(model, data_loader, epochs_per_step, lr=lr, log_freq=log_freq, loss=loss,
                  initial_epoch=(epochs_per_step * step), test_loader=val_loader)
            step += 1
        lr *= lr_mult
        log(f"\nLearning rate updated to {lr}")
