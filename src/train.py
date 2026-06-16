import config
import glob
import numpy as np
import os
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


def train_epoch(model, optimizer, train_loader, log_freq=1000, rng_piece_positions=False, base_loss=F.mse_loss,
                test_loader=None, name=None, writer=None, global_step=0):
    model.train()
    # Running sums of [total, reg, ce] kept on-device so we only synchronise (.item()) once
    # per log_freq batches instead of every batch -- cheaper than the previous per-batch
    # loss.item() while still exposing the loss components.
    epoch_sums = torch.zeros(3, device=config.device)
    recent_sums = torch.zeros(3, device=config.device)
    count = 0
    recent_count = 0
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
        optimizer.step()

        with torch.no_grad():
            batch_stats = torch.stack([total.detach(), reg.detach(), ce.detach()])
            epoch_sums += batch_stats
            recent_sums += batch_stats
        count += 1
        recent_count += 1
        global_step += 1
        if count % log_freq == 0:
            recent = (recent_sums / recent_count).tolist()  # single host sync
            total_avg = (epoch_sums[0] / count).item()
            batch_res_str = (f"Batch {count} Recent:{recent[0]:.6f} (reg {recent[1]:.6f}, ce {recent[2]:.6f}), "
                             f"Total:{total_avg:.6f}")
            if test_loader is not None:
                if config.rec is not None and config.rec > 1:
                    batch_res_str += ", " + gen_validation_string(model, test_loader, rec=1)
                batch_res_str += ", " + gen_validation_string(model, test_loader)
            log(batch_res_str)
            if writer is not None:
                writer.add_scalar("train/loss", recent[0], global_step)
                writer.add_scalar("train/loss_reg", recent[1], global_step)
                writer.add_scalar("train/loss_ce", recent[2], global_step)
            if name is not None:
                save(model, name=name)
            recent_sums = torch.zeros(3, device=config.device)
            recent_count = 0
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


def _move_optimizer_state(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def scheduled_lr_train(model, data_loader=None, val_loader=None, loss=F.mse_loss, init_lr=0.001, min_lr=0.0001,
                       lr_mult=0.5, epochs_per_step=1, log_freq=100000, resume_state=None, writer=None,
                       data_loader_fn=None, reload_every=0):
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

    epoch = start_epoch
    optimizer = None
    cur_step = None
    while True:
        step = epoch // epochs_per_step
        lr = init_lr * (lr_mult ** step)
        if lr < min_lr:
            break
        # Recreate the optimizer at each step boundary (matching the original schedule's
        # fresh-optimizer-per-lr behaviour), restoring saved state when resuming mid-step.
        if step != cur_step:
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
            if pending_opt_state is not None and pending_opt_step == step:
                optimizer.load_state_dict(pending_opt_state)
                _move_optimizer_state(optimizer, config.device)
                log(f"Restored optimizer state for step {step}")
            pending_opt_state = None
            cur_step = step
            log(f"\nLearning rate is {lr:g} (step {step})")
            if writer is not None:
                writer.add_scalar("train/lr", lr, global_step)
        # In resampling mode, draw a fresh random subset every reload_every epochs (and on
        # the first epoch of a (re)started run, where data_loader has not been built yet).
        if data_loader_fn is not None and (data_loader is None or epoch % reload_every == 0):
            data_loader = data_loader_fn()
        log(f"Epoch {epoch + 1}--Training on {len(data_loader.dataset)} samples"
            "----------------------------------------------------")
        _, global_step = train_epoch(model, optimizer, data_loader, log_freq=log_freq, base_loss=loss,
                                     test_loader=val_loader, name=config.name, writer=writer,
                                     global_step=global_step)
        epoch += 1
        save(model, f"../models/{config.name}/{config.name}_ep{epoch}")
        save_training_state(config.name, optimizer, next_epoch=epoch, step=step, global_step=global_step)
        if val_loader is not None:
            val_losses = test(model, val_loader, base_loss=[F.mse_loss, F.l1_loss])
            log(f"Finished Epoch {epoch}. Val mse:{val_losses[0]:.6f}, Val l1:{val_losses[1]:.6f}")
            if writer is not None:
                writer.add_scalar("val/mse", val_losses[0], global_step)
                writer.add_scalar("val/l1", val_losses[1], global_step)


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
