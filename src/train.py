import chess
import numpy as np
import os
import torch
import torch.nn.functional as F

from data import get_features


def loss_f(pred, target, base_loss=F.mse_loss):
    return base_loss(pred[:, 0] - pred[:, 2], 1.0 - target)


def loss_combined(pred_logit, target, base_loss=F.mse_loss):
    pred = F.softmax(pred_logit, dim=-1)
    return loss_f(pred, target, base_loss) + 0.04 * F.cross_entropy(pred_logit, target)


# def loss_custom_ce(pred, target):
# pred[:,1] = 1-torch.abs(pred[:,0]-pred[:,2])
# target = F.one_hot(target, num_classes=3)
# return F.binary_cross_entropy(pred, target, weight=custom_ce_loss_weights)


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


def get_board_from_tensor(board_tensor):
    assert len(board_tensor.shape) == 1 and board_tensor.shape[0] == 772
    board = chess.Board(chess960=True)
    board.clear()
    for square in range(64):
        for piecetype in range(12):
            if board_tensor[piecetype * 64 + square] != 0:
                board.set_piece_at(square, chess.Piece(1 + (piecetype % 6), 1 - piecetype // 6))
    return board


def get_boards_and_targets(loader, max_count=100):
    boards_and_targets = []
    for (data, target) in loader:
        for i in range(data.shape[0]):
            boards_and_targets.append((get_board_from_tensor(data[i]), target[i:i + 1], data[i:i + 1].detach()))
            if len(boards_and_targets) >= max_count:
                return boards_and_targets
    return boards_and_targets


def get_startpos_tensor():
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    feat, res = get_features(fen, "1-0", cond_h_flip=False)
    dense = np.squeeze(np.asarray(feat.todense()))
    return torch.Tensor(dense).view(1, 772)


def get_startpos_eval(model):
    training = model.training
    model.eval()
    with torch.no_grad():
        pred = model(get_startpos_tensor())[0]
    model.train(training)
    return pred, 0.5 + (pred[0] - pred[2]) / 2


def test(model, test_loader, base_loss=F.mse_loss):
    if not isinstance(base_loss, list):
        base_loss = [base_loss]
    training = model.training
    model.eval()
    loss_sum = np.zeros(len(base_loss))
    count = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            output = model(data.type(torch.float32))
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


def gen_validation_string(model, validation_loader):
    test_loss = test(model, validation_loader, base_loss=[F.mse_loss, F.l1_loss])
    start_pred, start_eval = get_startpos_eval(model)
    wdl_str = f"({start_pred[0]:.4f}/{start_pred[1]:.4f}/{start_pred[2]:.4f})"
    return f"Val mse:{test_loss[0]:.6f}, Val l1:{test_loss[1]:.6f}, Start WDL:{wdl_str}=>{start_eval:.5f}"


def train_epoch(model, optimizer, train_loader, log_freq=1000, rng_piece_positions=False, base_loss=F.mse_loss,
                test_loader=None):
    model.train()
    loss_sum = 0
    recent_loss = 0
    count = 0
    custom_ce_loss_weights = torch.Tensor([1.0, 0.2, 1.0])
    for (data, target) in train_loader:
        data = randomize_piece_positions(data, rng_piece_positions)
        optimizer.zero_grad()

        # output = model(data.type(torch.float32), activate=True)
        # loss = F.binary_cross_entropy(output, F.one_hot(target, num_classes=3).type(torch.float32))#, weight=custom_ce_loss_weights)

        output = model(data.type(torch.float32), activate=False)
        # loss = F.cross_entropy(output, target)
        loss = loss_combined(output, target, base_loss=base_loss)

        # output = model(data.type(torch.float32))
        # loss = loss_f(output, target, base_loss=base_loss)

        # loss = F.nll_loss(output[:,0::2], target[:,0::2])
        loss_sum += loss.item()
        recent_loss += loss.item()
        count += 1
        loss.backward()
        optimizer.step()
        if count % log_freq == 0:
            batch_res_str = f"Batch {count} Recent:{(recent_loss / log_freq):.6f}, Total:{(loss_sum / count):.6f}"
            if test_loader is not None:
                batch_res_str += ", " + gen_validation_string(model, test_loader)
            print(batch_res_str)
            recent_loss = 0
    return loss_sum / count


def train(model, name, train_loader, epochs, optimizer=None, lr=0.01, log_freq=100000, rng_piece_positions=False,
          loss=F.mse_loss, initial_epoch=0, test_loader=None):
    if optimizer is None:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    if not os.path.exists(f"Models/{name}"):
        os.makedirs(f"Models/{name}")
    for epoch in range(initial_epoch, initial_epoch + epochs):
        if epoch != initial_epoch:
            print()
        print(
            f"Epoch {epoch + 1}--Training on {len(train_loader.dataset)} samples----------------------------------------------------")
        train_epoch(model, optimizer, train_loader, log_freq=log_freq, rng_piece_positions=rng_piece_positions,
                    base_loss=loss, test_loader=test_loader)
        torch.save(model.state_dict(), f"Models/{name}/{name}_ep{epoch + 1}.pt")
        model.serialize(f"Models/{name}/{name}_ep{epoch + 1}.bin", verbose=1)
        if test_loader is not None:
            print(f"Finished Epoch {epoch + 1}. {gen_validation_string(model, test_loader)}")


def load_weights(model, name, ep):
    model.load_state_dict(torch.load(f"Models/{name}/{name}_ep{ep}.pt"))


def load_partial_model_weights(model, helper_model, n):
    with torch.no_grad():
        model.l1.weight[:n] = helper_model.l1.weight
        model.l1.bias[:n] = helper_model.l1.bias
        model.out.weight[:, :n] = helper_model.out.weight
        model.out.bias[:n] = helper_model.out.bias


def scheduled_lr_train(model, model_save, data_loader, val_loader=None, init_lr=0.001, min_lr=0.0001, lr_mult=0.5,
                       epochs_per_step=1, log_freq=100000):
    assert 1 > lr_mult > 0, f"Unexpected lr_mult param:{lr_mult}"
    step = 0
    lr = init_lr
    while lr >= min_lr:
        train(model, model_save, data_loader, epochs_per_step, lr=lr, log_freq=log_freq,
              initial_epoch=(epochs_per_step * step), test_loader=val_loader)
        lr *= lr_mult
        step += 1
        print(f"\nLearning rate updated to {lr}")
