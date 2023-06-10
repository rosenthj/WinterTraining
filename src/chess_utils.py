import chess
import numpy as np
import torch

from src.data import get_features
from utils import string_to_result_class, flip_result


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


def tb_probe_result(board):
    assert isinstance(board, chess.Board)
    assert board.turn == chess.WHITE
    with chess.syzygy.open_tablebase("syzygy") as tablebase:
        tb_res = tablebase.probe_wdl(board)
        if tb_res == 2:
            return 0
        elif tb_res == -2:
            return 2
        else:
            assert tb_res == 0 or tb_res == -1 or tb_res == 1
            return 1


def get_standardised_board_and_result(fen, result_str, cond_h_flip=False, cond_v_flip=False):
    board = chess.Board()
    board.set_fen(fen)
    hmc = board.halfmove_clock
    result = string_to_result_class(result_str)
    if board.turn == chess.BLACK:
        board = board.mirror()
        result = flip_result(result)
    if board.castling_rights == 0 and cond_h_flip:
        board = board.transform(chess.flip_horizontal)
    if board.pawns == 0 and cond_v_flip:
        board = board.transform(chess.flip_vertical)
    if len(board.piece_map()) <= 5:
        assert hmc == board.halfmove_clock
        result = tb_probe_result(board)
    return board, result


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