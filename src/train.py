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


def loss_components(pred_logit, target, base_loss=F.mse_loss, ce_weight=0.04, draw_weight=0.0):
    """Return ``(total, reg, ce, draw)``: the optimized loss and its three components.

    - ``reg``  -- WDL regression term: ``base_loss`` on the win-minus-loss probability. This is
      the eval axis (it drives playing strength) but it is *blind to draws* -- moving probability
      symmetrically between win and loss leaves the win-minus-loss margin unchanged.
    - ``ce``   -- categorical cross-entropy over W/D/L (a log-barrier term: sensitive to the
      *tails*, i.e. it punishes assigning near-zero probability to the realised outcome).
    - ``draw`` -- MSE of the draw probability against the draw indicator. This covers exactly the
      axis ``reg`` is blind to, with a bounded quadratic penalty. ``reg + 3*draw`` reproduces the
      multiclass Brier score (up to an overall scale), so ``draw_weight=3`` is "Brier"; smaller
      values keep the eval axis dominant.

    ``total = reg + ce_weight*ce + draw_weight*draw`` is what is optimized. The historical default
    (``ce_weight=0.04``, ``draw_weight=0``) is unchanged. ``ce`` and ``draw`` are always computed
    (both cheap) so they can be logged as diagnostics even when their weight is 0.
    """
    pred = F.softmax(pred_logit, dim=-1)
    reg = loss_f(pred, target, base_loss)
    ce = F.cross_entropy(pred_logit, target)
    draw = F.mse_loss(pred[:, 1], (target == 1).to(pred.dtype))  # p_draw vs draw indicator
    total = reg + ce_weight * ce + draw_weight * draw
    return total, reg, ce, draw


def loss_combined(pred_logit, target, base_loss=F.mse_loss, ce_weight=0.04, draw_weight=0.0):
    return loss_components(pred_logit, target, base_loss, ce_weight, draw_weight)[0]


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


def _activation_stats_dict(elem, agg, piece_sum, pos_count, max_val, eps):
    """Build the activation-health TensorBoard tags from the accumulators in ``test()``.

    Per-element saturation (``act/*``): fraction of clipped-ReLU values pinned at 0 / max,
    split into the conv feature map and the fc hidden layer. ``act/conv_active_frac_zero``
    corrects the conv zero fraction for the piece-presence mask -- most conv entries are
    structurally 0 at empty squares; the masked fraction is ``1 - pieces/768`` (the conv
    repeat_interleave/mirror/cat preserve density), so this isolates genuine clamping among
    the computed entries.

    Per-neuron deadness (``val/*``): channels/units that never rise above 0 (``*_dead_zero``)
    or are always pinned at the max (``*_dead_max``) across the whole validation set. Per-neuron
    aggregation over many positions is not masking-confounded (every square gets occupied
    somewhere), so a never-positive channel is genuinely dead.
    """
    stats = {}
    for bucket, (num_zero, num_max, total) in elem.items():
        if total:
            stats[f"act/{bucket}_frac_zero"] = num_zero / total
            stats[f"act/{bucket}_frac_max"] = num_max / total
    if pos_count and elem["conv"][2]:
        masked_frac = 1.0 - piece_sum / (pos_count * 768)
        if masked_frac < 1.0:
            frac_zero = elem["conv"][0] / elem["conv"][2]
            active = (frac_zero - masked_frac) / (1.0 - masked_frac)
            stats["act/conv_active_frac_zero"] = min(1.0, max(0.0, active))
    for bucket, (run_max, run_min) in agg.items():
        stats[f"val/{bucket}_dead_zero"] = (run_max <= eps).float().mean().item()
        stats[f"val/{bucket}_dead_max"] = (run_min >= max_val - eps).float().mean().item()
    return stats


def test(model, test_loader, base_loss=F.mse_loss, rec=None, return_accuracy=False, activation_stats=False,
         return_wasserstein=False):
    """Evaluate on ``test_loader`` in one pass. Optionally also returns argmax W/D/L accuracy,
    the Wasserstein-1 distance, and a dict of activation-health stats (see
    ``_activation_stats_dict``), all computed in the same forward pass via an activation hook --
    so the extra metrics cost no extra forward.

    Returns ``losses`` alone, or a tuple appending the requested extras in the order
    accuracy, wasserstein, activation_stats.
    """
    if not isinstance(base_loss, list):
        base_loss = [base_loss]
    training = model.training
    model.eval()
    loss_sum = np.zeros(len(base_loss))
    count = 0
    correct = 0
    n = 0
    w1_sum = 0.0
    # W/D/L outcomes live on the expected-score axis (White win 1.0, draw 0.5, Black win 0.0).
    # The target is a single outcome (a point mass), so the Wasserstein-1 distance from the
    # predicted distribution to it is just E_p|score - true_score|. Unlike the mse/l1 metrics
    # -- which only see the win-minus-loss margin -- this is sensitive to where the *draw*
    # mass sits, so it reflects draw calibration.
    w1_scores = torch.tensor([1.0, 0.5, 0.0], device=config.device)

    # Optional activation-health accumulation, gathered during this same pass via a hook.
    act = getattr(model, "activation", None)
    do_act = activation_stats and isinstance(act, torch.nn.Module)
    eps = 1e-4
    max_val = getattr(act, "max_val", 8.0) if do_act else 8.0
    elem = {"conv": [0, 0, 0], "fc": [0, 0, 0]}  # [num_zero, num_max, total]
    agg = {}  # bucket -> [running_max_per_unit, running_min_per_unit]
    piece_sum = 0.0
    pos_count = 0

    def hook(module, inputs, output):
        bucket = "fc" if output.dim() <= 2 else "conv"
        e = elem[bucket]
        e[0] += int((output <= eps).sum().item())
        e[1] += int((output >= max_val - eps).sum().item())
        e[2] += output.numel()
        dims = [d for d in range(output.dim()) if d != 1]  # reduce all but the channel/unit dim
        cur_max = output.amax(dim=dims)
        cur_min = output.amin(dim=dims)
        if bucket not in agg:
            agg[bucket] = [cur_max.clone(), cur_min.clone()]
        else:
            torch.maximum(agg[bucket][0], cur_max, out=agg[bucket][0])
            torch.minimum(agg[bucket][1], cur_min, out=agg[bucket][1])

    handle = act.register_forward_hook(hook) if do_act else None
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(config.device), target.to(config.device)
            if do_act:
                piece_sum += float(data[:, :768].sum().item())
                pos_count += data.shape[0]
            if rec is None:
                output = model(data.type(torch.float32).to(config.device))
            else:
                output = model(data.type(torch.float32).to(config.device), rec=rec)
            for i in range(len(base_loss)):
                loss = loss_f(output, target, base_loss=base_loss[i])
                loss_sum[i] += loss.item()
            if return_wasserstein:
                true_score = 1.0 - 0.5 * target.to(w1_scores.dtype)  # class 0/1/2 -> 1.0/0.5/0.0
                dist = (w1_scores.unsqueeze(0) - true_score.unsqueeze(1)).abs()  # (B, 3)
                w1_sum += (output * dist).sum(dim=1).mean().item()
            if return_accuracy:
                # output is the softmax over (white-win, draw, black-win); the target is the
                # result class, so argmax == target is a correct W/D/L call.
                correct += (output.argmax(dim=1) == target).sum().item()
                n += target.numel()
            count += 1
    if handle is not None:
        handle.remove()
    model.train(training)

    losses = loss_sum / count
    if not (return_accuracy or return_wasserstein or activation_stats):
        return losses
    result = [losses]
    if return_accuracy:
        result.append(correct / n if n else 0.0)
    if return_wasserstein:
        result.append(w1_sum / count if count else 0.0)
    if activation_stats:
        result.append(_activation_stats_dict(elem, agg, piece_sum, pos_count, max_val, eps))
    return tuple(result)


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


def train_epoch(model, optimizer, train_loader, log_freq=1000, rng_piece_positions=False, base_loss=F.mse_loss,
                test_loader=None, name=None, writer=None, global_step=0, lr=None, clip_grad_norm=None,
                ce_weight=0.04, draw_weight=0.0):
    model.train()
    # Running sums of [total, reg, ce, draw] kept on-device so we only synchronise (.item()) once
    # per log_freq batches instead of every batch -- cheaper than the previous per-batch
    # loss.item() while still exposing the loss components.
    epoch_sums = torch.zeros(4, device=config.device)
    recent_sums = torch.zeros(4, device=config.device)
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
        msg = (f"Batch {count} Recent:{recent[0]:.6f} (reg {recent[1]:.6f}, ce {recent[2]:.6f}, "
               f"draw {recent[3]:.6f}), Total:{total_avg:.6f}")
        val_losses = None
        val_acc = None
        val_w1 = None
        act_stats = None
        if include_val and test_loader is not None:
            if config.rec is not None and config.rec > 1:
                msg += ", " + gen_validation_string(model, test_loader, rec=1)
            # One validation pass yields losses, accuracy, Wasserstein and activation-health stats.
            val_losses, val_acc, val_w1, act_stats = test(model, test_loader, base_loss=[F.mse_loss, F.l1_loss],
                                                          return_accuracy=True, return_wasserstein=True,
                                                          activation_stats=True)
            msg += (f", Val mse:{val_losses[0]:.6f}, Val l1:{val_losses[1]:.6f}, "
                    f"Val w1:{val_w1:.6f}, Val acc:{val_acc:.4f}")
        log(msg)
        if writer is not None:
            writer.add_scalar("train/loss", recent[0], global_step)
            writer.add_scalar("train/loss_reg", recent[1], global_step)
            writer.add_scalar("train/loss_ce", recent[2], global_step)
            writer.add_scalar("train/loss_draw", recent[3], global_step)
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
                writer.add_scalar("val/wasserstein", val_w1, global_step)
                writer.add_scalar("val/accuracy", val_acc, global_step)
                for tag, frac in act_stats.items():
                    writer.add_scalar(tag, frac, global_step)
        if name is not None:
            save(model, name=name)
        recent_sums = torch.zeros(4, device=config.device)
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
        total, reg, ce, draw = loss_components(output, target, base_loss=base_loss,
                                               ce_weight=ce_weight, draw_weight=draw_weight)
        total.backward()
        if clip_grad_norm:
            # clip_grad_norm_ returns the total norm *before* clipping (kept on-device; only
            # synced to host at log time) and rescales the gradients in place.
            last_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        optimizer.step()

        with torch.no_grad():
            batch_stats = torch.stack([total.detach(), reg.detach(), ce.detach(), draw.detach()])
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


def save(model, path=None, name=None, epoch=None, write_bin=None):
    # The Winter-readable .bin is only kept for the _tmp checkpoint (the always-most-recent
    # latest pointer, refreshed every flush) -- it's a convenience for loading into the engine.
    # Per-epoch snapshots store only the .pt; a .bin for any of them can be regenerated later
    # via model.serialize(). write_bin overrides this default when given.
    is_tmp = path is None and epoch is None
    if path is None:
        assert name is not None
        if epoch is None:
            path = f"../models/{name}/{name}_tmp"
        else:
            path = f"../models/{name}/{name}_ep{epoch + 1}"
    torch.save(model.state_dict(), f"{path}.pt")
    if write_bin if write_bin is not None else is_tmp:
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


def load_partial_state_dict(model, state_dict, verbose=True):
    """Seed ``model`` from ``state_dict``, copying the overlapping slice of every shared
    parameter and leaving the rest at its fresh initialization.

    This is the generic ("grow the net") replacement for the old, ``Net``-specific
    ``load_partial_model_weights``: for each parameter present in both the checkpoint and
    the target model, it copies ``src[:min(d0), :min(d1), ...]`` into the target, so a
    checkpoint trained at a smaller ``--d`` / ``--fd`` seeds the leading rows/columns of a
    larger model and the newly added units start from their normal random init. Parameters
    absent from the checkpoint (e.g. a whole new layer) are left untouched.

    Returns (copied, grown, skipped) parameter-name lists for logging.
    """
    tgt = model.state_dict()
    copied, grown, skipped = [], [], []
    with torch.no_grad():
        for name, src in state_dict.items():
            if name not in tgt:
                skipped.append(name)
                continue
            dst = tgt[name]
            if src.shape == dst.shape:
                dst.copy_(src)
                copied.append(name)
            elif src.dim() == dst.dim() and all(s <= d for s, d in zip(src.shape, dst.shape)):
                # Copy the overlapping leading block; the rest keeps its fresh init.
                sl = tuple(slice(0, s) for s in src.shape)
                dst[sl].copy_(src)
                grown.append(name)
            else:
                # Shrinking or a rank change we can't safely align -- leave it initialized.
                skipped.append(name)
    model.load_state_dict(tgt)
    if verbose:
        print(f"Partial load: {len(copied)} copied, {len(grown)} grown "
              f"{grown if grown else ''}, {len(skipped)} skipped "
              f"{skipped if skipped else ''}")
    return copied, grown, skipped


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
                       lookahead_alpha=0.5, reg_weights_only=False, ce_weight=0.04, draw_weight=0.0):
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
                                     global_step=global_step, lr=lr, clip_grad_norm=clip_grad_norm,
                                     ce_weight=ce_weight, draw_weight=draw_weight)
        epoch += 1
        save(model, f"../models/{config.name}/{config.name}_ep{epoch}")
        save_training_state(config.name, optimizer, next_epoch=epoch, step=step, global_step=global_step)
        if val_loader is not None:
            val_losses, val_acc, val_w1, act_stats = test(model, val_loader, base_loss=[F.mse_loss, F.l1_loss],
                                                          return_accuracy=True, return_wasserstein=True,
                                                          activation_stats=True)
            log(f"Finished Epoch {epoch}. Val mse:{val_losses[0]:.6f}, Val l1:{val_losses[1]:.6f}, "
                f"Val w1:{val_w1:.6f}, Val acc:{val_acc:.4f}")
            if writer is not None:
                writer.add_scalar("val/mse", val_losses[0], global_step)
                writer.add_scalar("val/l1", val_losses[1], global_step)
                writer.add_scalar("val/wasserstein", val_w1, global_step)
                writer.add_scalar("val/accuracy", val_acc, global_step)
                for tag, frac in act_stats.items():
                    writer.add_scalar(tag, frac, global_step)
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
