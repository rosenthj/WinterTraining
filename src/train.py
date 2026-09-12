import config
import glob
import math
import numpy as np
import os
import signal
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
                ce_weight=0.04, draw_weight=0.0, lr_schedule=None, max_batches=None, budget=None,
                checkpoint_fn=None):
    """Train for one pass (or ``max_batches`` batches) and return
    ``(avg_loss, global_step, batches_trained, stop_reason)``.

    ``lr_schedule``: optional callable global_step -> lr. When given, the LR is set per batch
    from the current global_step (used by the WSD schedule for step-granular warmup/decay);
    when None, the fixed ``lr`` set by the caller is used for the whole epoch (legacy behaviour).

    ``max_batches`` caps the pass, which is how a partial epoch is resumed: a job that died
    partway through epoch N runs only the batches N had left.

    ``budget`` (a :class:`SegmentBudget`) and ``checkpoint_fn`` make the pass interruptible.
    Both are polled between optimizer steps -- the only point at which the weights, the
    optimizer state and the step counters all describe the same position in the epoch -- so
    ``checkpoint_fn(batches_trained, global_step)`` always captures a consistent point.
    ``stop_reason`` is non-None when the pass ended because the budget said to stop rather
    than because the data (or ``max_batches``) ran out.
    """
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

    stop_reason = None
    for (data, target) in train_loader:
        if max_batches is not None and count >= max_batches:
            break
        # data = randomize_piece_positions(data, rng_piece_positions)
        if lr_schedule is not None:
            # Per-step LR (warmup/stable/decay). Reassigning the local ``lr`` also updates what
            # flush() logs to train/lr, so the curve follows the schedule at log resolution.
            lr = lr_schedule(global_step)
            _set_lr(optimizer, lr)
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
        # Between steps: take a periodic checkpoint, and stop here if the job is ending.
        if budget is not None:
            stop_reason = budget.stop_reason()
            if stop_reason is not None or budget.due_for_checkpoint():
                if checkpoint_fn is not None:
                    checkpoint_fn(count, global_step)
                budget.mark_checkpoint()
            if stop_reason is not None:
                break
    flush(include_val=False)  # capture the tail of the epoch (no extra validation pass)
    avg_loss = (epoch_sums[0] / count).item() if count else 0.0
    return avg_loss, global_step, count, stop_reason


def save(model, path=None, name=None, epoch=None, write_bin=None):
    # The Winter-readable net is only kept for the _tmp checkpoint (the always-most-recent
    # latest pointer, refreshed every flush) -- it's a convenience for loading into the engine.
    # Per-epoch snapshots store only the .pt; a net for any of them can be regenerated later
    # via model.serialize_quantized(). write_bin overrides this default when given.
    is_tmp = path is None and epoch is None
    if path is None:
        assert name is not None
        if epoch is None:
            path = f"../models/{name}/{name}_tmp"
        else:
            path = f"../models/{name}/{name}_ep{epoch + 1}"
    torch.save(model.state_dict(), f"{path}.pt")
    if write_bin if write_bin is not None else is_tmp:
        # Winter reads the quantized .qbin. serialize() is kept for the architectures
        # that do not have a quantized exporter, and for anything wanting raw floats.
        if hasattr(model, "serialize_quantized"):
            model.serialize_quantized(f"{path}.qbin", verbose=1)
        else:
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


def resolve_deadline(max_seconds=None):
    """Return the unix time at which this job's allocation ends, or None if unknown.

    Prefers SLURM's own ``SLURM_JOB_END_TIME`` (exported for every job, array tasks
    included) so nothing has to be kept in sync with the ``#SBATCH --time`` line. A
    ``max_seconds`` budget, when given, wins: it is the explicit override for running
    outside SLURM or on a version that does not export the end time.
    """
    if max_seconds is not None and max_seconds > 0:
        return time.time() + max_seconds
    raw = os.environ.get("SLURM_JOB_END_TIME")
    if not raw:
        return None
    try:
        end = float(raw)
    except ValueError:
        log(f"Ignoring unparseable SLURM_JOB_END_TIME={raw!r}; no wall-clock deadline.")
        return None
    if end <= time.time():
        log(f"Ignoring SLURM_JOB_END_TIME={raw!r} (not in the future); no wall-clock deadline.")
        return None
    return end


class SegmentBudget:
    """Decides when a training segment should checkpoint, and when it should stop.

    One object answers both questions the batch loop polls, because they are two halves of
    the same problem -- a run split into preemptable ~4h segments loses whatever work sits
    between the last checkpoint and the segment's death:

    - *Announced* deaths (the wall-clock limit approaching, or a SIGTERM from a preemption's
      grace period / a ``--signal`` request) set the stop flag, and the loop checkpoints and
      exits cleanly at the next batch boundary. Cost: one batch.
    - *Unannounced* deaths (a preemption that outruns its grace period, a node failure)
      can't be caught at all, so ``checkpoint_every_mins`` bounds what they destroy.

    The deadline is authoritative and needs no cooperation from the scheduler; signals are
    the belt-and-braces path for when a job ends early or ``--signal`` delivery differs
    between SLURM configurations.
    """

    def __init__(self, deadline=None, reserve_seconds=120.0, checkpoint_every_mins=10.0):
        self.deadline = deadline
        self.reserve_seconds = max(0.0, reserve_seconds)
        self.checkpoint_every = max(0.0, checkpoint_every_mins) * 60.0
        self.signalled = None
        self._last_checkpoint = time.monotonic()

    def install_signal_handlers(self):
        # SIGINT is deliberately left alone so Ctrl-C keeps aborting immediately.
        for sig in (signal.SIGTERM, signal.SIGUSR1):
            try:
                signal.signal(sig, self._handle)
            except (ValueError, OSError) as e:  # non-main thread, or unsupported platform
                log(f"Could not install a {sig.name} handler ({e}); "
                    "relying on the wall-clock deadline alone.")
        return self

    def _handle(self, signum, frame):
        # Record the request only. Checkpointing from a signal handler could land in the
        # middle of a backward pass or of another torch.save; the loop picks this up at the
        # next batch boundary instead.
        if self.signalled is None:
            self.signalled = signal.Signals(signum).name

    def seconds_left(self):
        return None if self.deadline is None else self.deadline - time.time()

    def stop_reason(self):
        """A short description of why training should stop now, or None to keep going."""
        if self.signalled is not None:
            return f"received {self.signalled}"
        left = self.seconds_left()
        if left is not None and left <= self.reserve_seconds:
            return f"wall-clock limit in {max(0.0, left):.0f}s"
        return None

    def due_for_checkpoint(self):
        return (self.checkpoint_every > 0
                and time.monotonic() - self._last_checkpoint >= self.checkpoint_every)

    def mark_checkpoint(self):
        self._last_checkpoint = time.monotonic()

    def describe(self):
        left = self.seconds_left()
        when = "no wall-clock deadline" if left is None else f"{left / 60:.1f} min of wall clock left"
        cadence = ("mid-epoch checkpointing off" if self.checkpoint_every <= 0
                   else f"checkpointing every {self.checkpoint_every / 60:g} min")
        return (f"{when}, stopping {self.reserve_seconds:g}s before it, {cadence}")


def training_state_path(name):
    return f"../models/{name}/{name}.state.pt"


def save_training_state(name, optimizer, next_epoch, step, global_step=0, model=None,
                        batches_done=0, steps_per_epoch=None):
    """Persist everything a later job needs to resume, as one atomic file.

    The model weights go in this file too (``model``), so a resume reads a single mutually
    consistent object -- weights, optimizer momentum, schedule position, and the offset
    *within* the epoch. ``batches_done`` is 0 at an epoch boundary and the number of batches
    of epoch ``next_epoch`` already trained when the checkpoint was taken mid-epoch;
    ``steps_per_epoch`` records how long that epoch is, so the resume knows what is left.

    Written to a sibling temp path and moved into place with ``os.replace``, so a job killed
    during the write leaves the previous checkpoint intact instead of a truncated file --
    the difference between losing one checkpoint interval and losing the whole run.
    """
    state = {"next_epoch": next_epoch, "step": step, "global_step": global_step,
             "batches_done": batches_done, "steps_per_epoch": steps_per_epoch,
             "optimizer": optimizer.state_dict()}
    if model is not None:
        state["model"] = model.state_dict()
    path = training_state_path(name)
    tmp = f"{path}.writing"
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_training_state(name):
    """Return the saved schedule/optimizer state for ``name``, or None if absent."""
    path = training_state_path(name)
    if os.path.exists(path):
        return torch.load(path, map_location="cpu")
    return None


def latest_checkpoint(name):
    """Return the most recently modified weight checkpoint (.pt) for ``name``, or None.

    Matches ``{name}_*.pt`` (e.g. ``{name}_ep3.pt``, ``{name}_tmp.pt``) but not the
    ``{name}.state.pt`` schedule file. Prefer :func:`load_resume_state` for resuming: the
    newest file by mtime is not necessarily the one the saved schedule state describes.
    """
    paths = glob.glob(f"../models/{name}/{name}_*.pt")
    return max(paths, key=os.path.getmtime) if paths else None


def load_resume_state(name):
    """Return ``(weights, state)`` to resume run ``name`` from; either may be None.

    ``weights`` is a model state_dict and ``state`` the schedule/optimizer state that
    belongs with it. Checkpoints written by the current code carry the weights inside the
    state file, so the pair is consistent by construction.

    For a run whose state file predates that, the weights are taken from the per-epoch
    snapshot the state file names (``_ep{next_epoch}.pt``) rather than from the newest file
    on disk: ``_tmp.pt`` is refreshed mid-epoch by the logging flush, so choosing by mtime
    could pair mid-epoch weights with an epoch-boundary optimizer and ``global_step``.
    """
    state = load_training_state(name)
    if state is not None and state.get("model") is not None:
        log(f"Resume: weights + optimizer from {training_state_path(name)}")
        return state.pop("model"), state
    if state is not None:
        snapshot = f"../models/{name}/{name}_ep{state.get('next_epoch', 0)}.pt"
        if os.path.exists(snapshot):
            log(f"Resume: weights from {snapshot} (pre-combined state file)")
            return torch.load(snapshot, map_location="cpu"), state
        log(f"Resume: {snapshot} is missing; falling back to the newest checkpoint.")
    ckpt = latest_checkpoint(name)
    if ckpt is None:
        return None, state
    log(f"Resume: weights from {ckpt}")
    return torch.load(ckpt, map_location="cpu"), state


def _copy_overlap(dst, src, blocks=None):
    """Copy ``src`` into the overlapping leading region of ``dst`` (same rank, ``src`` no
    larger along any dim). If ``blocks`` is given as ``(dim, n)``, both tensors are split
    into ``n`` equal chunks along ``dim`` first and the leading block is copied per chunk --
    this keeps block-concatenated axes (e.g. NetRelHD's ``[real | mirror]`` channel axis)
    aligned when a shared dimension grows, instead of copying one contiguous prefix that
    would slice across the block boundary.
    """
    if blocks is None:
        sl = tuple(slice(0, s) for s in src.shape)
        dst[sl].copy_(src)
        return
    dim, n = blocks
    assert src.shape[dim] % n == 0 and dst.shape[dim] % n == 0, \
        f"block dim {dim} not divisible by {n}"
    for s_chunk, d_chunk in zip(src.chunk(n, dim=dim), dst.chunk(n, dim=dim)):
        sl = tuple(slice(0, s) for s in s_chunk.shape)
        d_chunk[sl].copy_(s_chunk)


def load_partial_state_dict(model, state_dict, verbose=True):
    """Seed ``model`` from ``state_dict``, copying the overlapping slice of every shared
    parameter and leaving the rest at its fresh initialization.

    This is the generic ("grow the net") replacement for the old, ``Net``-specific
    ``load_partial_model_weights``: for each parameter present in both the checkpoint and
    the target model, it copies ``src[:min(d0), :min(d1), ...]`` into the target, so a
    checkpoint trained at a smaller ``--d`` / ``--fd`` seeds the leading rows/columns of a
    larger model and the newly added units start from their normal random init. Parameters
    absent from the checkpoint (e.g. a whole new layer) are left untouched.

    Models with block-concatenated axes (e.g. NetRelHD, whose ``out``/``fout`` inputs are a
    ``[real | mirror]`` concatenation) can expose ``partial_load_blocks()`` -> ``{param_name:
    (dim, n_blocks)}`` so those axes are grown block-wise rather than as one contiguous prefix.

    Returns (copied, grown, skipped) parameter-name lists for logging.
    """
    tgt = model.state_dict()
    block_spec = getattr(model, "partial_load_blocks", lambda: {})()
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
                _copy_overlap(dst, src, block_spec.get(name))
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


def make_optimizer(name, params, lr, momentum=0.9, weight_decay=0.0):
    """Build the optimizer selected on the CLI: SGD with momentum, the only option."""
    name = name.lower()
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer '{name}' (expected 'sgd')")


def _set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group["lr"] = lr


def wsd_lr(step, total_steps, peak_lr, min_lr, warmup_steps, decay_steps):
    """Warmup-Stable-Decay learning rate at global ``step``.

    Three phases over a fixed budget of ``total_steps`` batches:
      - warmup  (``[0, warmup_steps)``): linear ramp 0 -> ``peak_lr``;
      - stable  (until the decay window): constant ``peak_lr``;
      - decay   (last ``decay_steps``): half-cosine ``peak_lr`` -> ``min_lr``.
    Unlike geometric step decay, the LR is held at its peak through the long stable phase and
    only annealed at the very end, so training keeps learning (helps rare features, which see
    few gradient steps) and most of the final loss drop comes from the short decay tail.
    Being a pure function of ``step`` (which is persisted as ``global_step``), it resumes exactly.
    """
    if warmup_steps > 0 and step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    decay_start = total_steps - decay_steps
    if step < decay_start:
        return peak_lr
    prog = min(1.0, max(0.0, (step - decay_start) / max(1, decay_steps)))
    return min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + math.cos(math.pi * prog))


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
    bias-like parameters from weight decay when ``reg_weights_only``."""
    if not reg_weights_only:
        return model.parameters()
    regularized, others = split_regularized_params(model)
    return [
        {"params": regularized},
        {"params": others, "weight_decay": 0.0},
    ]


def _move_optimizer_state(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def scheduled_lr_train(model, data_loader=None, val_loader=None, loss=F.mse_loss, init_lr=0.001, min_lr=0.0001,
                       lr_mult=0.5, epochs_per_step=1, log_freq=100000, resume_state=None, writer=None,
                       data_loader_fn=None, reload_every=0, optimizer_name="sgd", momentum=0.9,
                       weight_decay=0.0, persistent_optimizer=False,
                       clip_grad_norm=None, reg_weights_only=False, ce_weight=0.04, draw_weight=0.0,
                       schedule="step", total_epochs=None, warmup_steps=0, decay_frac=0.1,
                       budget=None):
    """Train with a learning-rate schedule.

    ``schedule`` selects the LR shape:
      - ``"step"`` (default): the historical geometric step decay -- ``lr = init_lr *
        lr_mult**(epoch // epochs_per_step)``, run until ``lr < min_lr``. Kept unchanged so
        older runs reproduce exactly.
      - ``"wsd"``: Warmup-Stable-Decay (``wsd_lr``), a step-granular schedule over a fixed
        ``total_epochs`` budget: linear warmup over ``warmup_steps`` batches to the peak
        ``init_lr``, a long stable phase at the peak, then a half-cosine decay to ``min_lr``
        over the final ``decay_frac`` of the run. Uses one persistent optimizer (continuous
        momentum). ``lr_mult`` / ``epochs_per_step`` are ignored.

    Either pass a fixed ``data_loader``, or pass ``data_loader_fn`` (a callable returning a
    fresh loader) together with ``reload_every`` > 0 to resample the training data every
    ``reload_every`` epochs. Resampling keeps only a subset of the corpus resident at once
    (see ``loader.load_from_multiple``'s ``portion``) while still covering all of it over
    time -- this is the memory-bounded streaming used for the full dataset.

    ``budget`` (a :class:`SegmentBudget`) makes the run interruptible at batch granularity
    rather than epoch granularity: it checkpoints periodically mid-epoch and, when the job's
    wall clock is nearly up or a stop signal arrives, saves and returns instead of being
    killed partway through an epoch. The next segment resumes the interrupted epoch from the
    batch it stopped on, so a segment boundary costs a checkpoint interval at worst instead
    of every batch since the last epoch boundary. Without a ``budget`` the behaviour is the
    old one: checkpoints at epoch boundaries only.
    """
    assert schedule in ("step", "wsd"), f"Unknown schedule '{schedule}' (expected 'step' or 'wsd')"
    wsd = schedule == "wsd"
    assert wsd or 1 > lr_mult > 0, f"Unexpected lr_mult param:{lr_mult}"
    assert not wsd or (total_epochs and total_epochs > 0), "WSD schedule requires total_epochs > 0"
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
    # Offset into the epoch we are resuming: 0 when the last checkpoint fell on an epoch
    # boundary, otherwise the number of batches of epoch start_epoch already trained.
    batches_done = 0
    resume_steps_per_epoch = None
    steps_per_epoch = None
    if resume_state is not None:
        start_epoch = resume_state.get("next_epoch", 0)
        global_step = resume_state.get("global_step", 0)
        pending_opt_state = resume_state.get("optimizer")
        pending_opt_step = resume_state.get("step")
        batches_done = resume_state.get("batches_done") or 0
        resume_steps_per_epoch = resume_state.get("steps_per_epoch")
        partway = f", {batches_done} batches into it" if batches_done else ""
        log(f"Resuming schedule at epoch {start_epoch + 1} (global_step {global_step}){partway}")
    if budget is not None:
        log(f"Segment budget: {budget.describe()}")

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

    # WSD sizes its warmup/stable/decay windows in batches, so it needs the per-epoch step
    # count. Build the first loader up front (in resampling mode) to measure it; the loop
    # reuses it for the first epoch. total_steps is a pure function of total_epochs and the
    # (constant) per-epoch step count, so wsd_lr(global_step) resumes exactly.
    lr_schedule = None
    # True when data_loader below is an unused fresh draw, so the first epoch trains on it
    # instead of immediately discarding it for another. Without this, a WSD run with
    # --reload-every pays two full reloads before its first batch of every segment.
    loader_unused = False
    if wsd:
        if data_loader is None:
            data_loader = data_loader_fn()
            loader_unused = data_loader_fn is not None
        steps_per_epoch = len(data_loader)
        total_steps = total_epochs * steps_per_epoch
        decay_steps = max(1, int(round(decay_frac * total_steps)))
        peak_lr = init_lr
        lr_schedule = lambda gs: wsd_lr(gs, total_steps, peak_lr, min_lr, warmup_steps, decay_steps)
        log(f"WSD schedule: {total_epochs} epochs x {steps_per_epoch} steps = {total_steps} steps "
            f"(warmup {warmup_steps}, decay {decay_steps}), peak lr {peak_lr:g} -> min {min_lr:g}")

    epoch = start_epoch
    optimizer = None
    cur_step = None
    # How long the last --reload-every draw took, so the loop can decline to start one it
    # hasn't time to finish (a reload of the full corpus is minutes of the allocation).
    last_reload_secs = None
    while True:
        if wsd:
            # Fixed-budget run; the LR is set per batch inside train_epoch from lr_schedule.
            if epoch >= total_epochs:
                break
            step = epoch  # stored for bookkeeping only (WSD is driven by global_step)
            lr = lr_schedule(global_step)  # epoch-start value for the log line below
            if optimizer is None:
                # One continuous optimizer for the whole WSD run (preserves momentum through
                # the stable->decay transition), so always restore saved state on resume.
                optimizer = make_optimizer(optimizer_name, _optimizer_params(model, reg_weights_only), lr,
                                           momentum=momentum, weight_decay=weight_decay)
                if pending_opt_state is not None:
                    optimizer.load_state_dict(pending_opt_state)
                    _move_optimizer_state(optimizer, config.device)
                    log("Restored optimizer state (WSD)")
                pending_opt_state = None
                log(f"\nWSD peak lr {peak_lr:g}; resuming at global_step {global_step} (epoch {epoch + 1})")
        else:
            step = epoch // epochs_per_step
            lr = init_lr * (lr_mult ** step)
            if lr < min_lr:
                break
            # Optimizer handling at each LR-schedule step:
            #  - default: recreate a fresh optimizer per step (the original behaviour);
            #  - persistent_optimizer: build it once and span the whole run, just updating the
            #    learning rate at each step -- this carries the momentum buffers across LR
            #    drops instead of resetting them.
            if optimizer is None:
                optimizer = make_optimizer(optimizer_name, _optimizer_params(model, reg_weights_only), lr,
                                           momentum=momentum, weight_decay=weight_decay)
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
        will_reload = (data_loader_fn is not None and not loader_unused
                       and (data_loader is None or epoch % reload_every == 0))
        loader_unused = False
        # Don't start work that cannot finish. Stopping here leaves the checkpoint on the
        # boundary it already sits on, and costs nothing; pushing on would spend the rest of
        # the allocation on a data reload, or on batches, that the wall clock then discards.
        if budget is not None:
            reason = budget.stop_reason()
            if reason is None and will_reload and last_reload_secs is not None:
                left = budget.seconds_left()
                if left is not None and left < last_reload_secs + budget.reserve_seconds:
                    reason = (f"{left / 60:.1f} min left, less than the "
                              f"{last_reload_secs / 60:.1f} min a data reload costs")
            if reason is not None:
                # batches_done is still whatever this epoch was resumed at (0 at a boundary),
                # and resume_steps_per_epoch the epoch length that went with it.
                save_training_state(config.name, optimizer, next_epoch=epoch, step=step,
                                    global_step=global_step, model=model,
                                    batches_done=batches_done,
                                    steps_per_epoch=resume_steps_per_epoch)
                log(f"Stopping before epoch {epoch + 1} ({reason}). "
                    "Checkpoint written; resume to continue.")
                return
        if will_reload:
            reload_t0 = time.perf_counter()
            # Drop the previous draw *before* building the next one. Holding the name while
            # data_loader_fn() runs keeps the old subset alive through the new one's
            # construction, so peak memory is the two draws plus vstack's copy rather than
            # one draw plus the copy -- the difference between fitting in --mem and not, at a
            # large --portion.
            data_loader = None
            data_loader = data_loader_fn()
            last_reload_secs = time.perf_counter() - reload_t0
            log(f"Drew a fresh training subset in {last_reload_secs:.0f}s")
        # An epoch is a *budget of batches*, not one specific enumeration of rows. When
        # batches_done > 0 the previous segment was interrupted partway through this epoch,
        # so only the remainder is trained now, drawn from a freshly shuffled loader. For
        # this data that is equivalent to finishing the interrupted pass -- the sampler
        # reshuffles every epoch anyway, and with --reload-every the subset is redrawn from
        # the corpus too -- while keeping the cost of epoch N at exactly epoch_steps batches
        # however often it is cut. That invariant is what keeps the WSD horizon
        # (total_epochs * steps_per_epoch) exact across interruptions.
        if resume_steps_per_epoch:
            epoch_steps = resume_steps_per_epoch
            resume_steps_per_epoch = None
        elif wsd:
            epoch_steps = steps_per_epoch  # the length the schedule's horizon was sized from
        else:
            epoch_steps = len(data_loader)
        remaining = max(0, epoch_steps - batches_done)
        log(f"Epoch {epoch + 1}--Training on {len(data_loader.dataset)} samples"
            "----------------------------------------------------")
        if batches_done:
            log(f"Continuing epoch {epoch + 1} at batch {batches_done}/{epoch_steps} "
                f"({remaining} to go)")

        def checkpoint_fn(trained_now, gs, _epoch=epoch, _step=step, _base=batches_done,
                          _epoch_steps=epoch_steps):
            # Mid-epoch: the schedule position is still epoch _epoch, with _base+trained_now
            # of its batches behind us. Defaults bind the loop's current values.
            save_training_state(config.name, optimizer, next_epoch=_epoch, step=_step,
                                global_step=gs, model=model,
                                batches_done=_base + trained_now, steps_per_epoch=_epoch_steps)

        # lr is logged densely inside train_epoch (at each log point + epoch end) so the
        # train/lr curve reads as a step function instead of an interpolated ramp. In WSD mode
        # lr_schedule overrides it per batch (step-granular warmup/decay).
        _, global_step, trained, stop_reason = train_epoch(
            model, optimizer, data_loader, log_freq=log_freq, base_loss=loss,
            test_loader=val_loader, name=config.name, writer=writer,
            global_step=global_step, lr=lr, clip_grad_norm=clip_grad_norm,
            ce_weight=ce_weight, draw_weight=draw_weight, lr_schedule=lr_schedule,
            max_batches=remaining, budget=budget, checkpoint_fn=checkpoint_fn)

        if stop_reason is not None and trained < remaining:
            # Interrupted mid-epoch. train_epoch checkpointed at the batch it stopped on, so
            # the next segment picks this epoch up from there; returning here skips the
            # per-epoch snapshot and validation, which belong to completed epochs only.
            log(f"Stopping {batches_done + trained}/{epoch_steps} batches into epoch "
                f"{epoch + 1} ({stop_reason}). Checkpoint written; resume to continue.")
            return
        batches_done = 0
        epoch += 1
        save(model, f"../models/{config.name}/{config.name}_ep{epoch}")
        save_training_state(config.name, optimizer, next_epoch=epoch, step=step,
                            global_step=global_step, model=model, batches_done=0,
                            steps_per_epoch=epoch_steps)
        if budget is not None:
            budget.mark_checkpoint()
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
        if stop_reason is not None:
            log(f"Stopping after epoch {epoch} ({stop_reason}). "
                "Checkpoint written; resume to continue.")
            return


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
