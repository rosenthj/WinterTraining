import chess
import chess.syzygy
import numpy as np
import torch
import scipy
import count
import config

from utils import string_to_result_class, flip_result


def get_board_from_tensor(board_tensor):
    assert len(board_tensor.shape) == 1 and board_tensor.shape[0] == 772, f"Tensor shape: {board_tensor.shape}"
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


def tb_res_to_wdl(tb_res):
    if tb_res == 2:
        return 0
    elif tb_res == -2:
        return 2
    else:
        assert tb_res == 0 or tb_res == -1 or tb_res == 1
        return 1


def tb_probe_result(board):
    assert isinstance(board, chess.Board)
    assert board.turn == chess.WHITE
    count.total_tb_queries[len(board.piece_map())] += 1
    if config.tablebase is not None:
        return tb_res_to_wdl(config.tablebase.probe_wdl(board))
    # with chess.syzygy.open_tablebase("./../syzygy") as tablebase:
    with chess.syzygy.open_tablebase("../../../Chess/TB_Merged") as tablebase:
        return tb_res_to_wdl(tablebase.probe_wdl(board))


def get_standardised_board_and_result(fen, result, cond_h_flip=False, cond_v_flip=False,
                                      get_white_perspective_result=False):
    board = chess.Board(chess960=True)
    board.set_fen(fen)
    hmc = board.halfmove_clock
    result = string_to_result_class(result)
    stm_is_black = (board.turn == chess.BLACK)
    if stm_is_black:
        board = board.mirror()
        result = flip_result(result)
    if board.castling_rights == 0 and cond_h_flip:
        board = board.transform(chess.flip_horizontal)
    if board.pawns == 0 and cond_v_flip:
        board = board.transform(chess.flip_vertical)
    if (len(board.piece_map()) <= 6 and not board.has_castling_rights(chess.WHITE)
            and not board.has_castling_rights(chess.BLACK)):
        assert hmc == board.halfmove_clock
        old_result = result
        result = tb_probe_result(board)
        if result != old_result:
            count.tb_changed[len(board.piece_map())] += 1
    if not get_white_perspective_result:
        return board, result
    if stm_is_black:
        return board, result, flip_result(result)
    return board, result, result


def get_board_tensor(board, sparse=False):
    features_board = np.zeros(2*6*64)
    features_castling = np.array([board.has_queenside_castling_rights(chess.WHITE), board.has_kingside_castling_rights(chess.WHITE),
                                     board.has_queenside_castling_rights(chess.BLACK), board.has_kingside_castling_rights(chess.BLACK)]).astype(np.int8)
    for square in board.piece_map():
        piece = board.piece_map()[square]
        idx = (piece.piece_type-1) * 64 + square
        if piece.color == chess.BLACK:
            idx += 6 * 64
        features_board[idx] = 1
    features = np.concatenate((features_board, features_castling)).astype(np.int8)
    assert features.shape[-1] == 772
    if sparse:
        return scipy.sparse.csr_matrix(features)
    return torch.Tensor(features).view(1, -1)


def get_features(fen, result_str, cond_h_flip=False, cond_v_flip=False, get_w_persp_result=False):
    board, result, wp_res = get_standardised_board_and_result(fen, result_str, cond_h_flip, cond_v_flip, True)
    result = torch.tensor([result])
    features_board = np.zeros(2*6*64)
    features_castling = np.array([board.has_queenside_castling_rights(chess.WHITE), board.has_kingside_castling_rights(chess.WHITE),
                                     board.has_queenside_castling_rights(chess.BLACK), board.has_kingside_castling_rights(chess.BLACK)]).astype(np.int8)
    for square in board.piece_map():
        piece = board.piece_map()[square]
        idx = (piece.piece_type-1) * 64 + square
        if piece.color == chess.BLACK:
            idx += 6 * 64
        features_board[idx] = 1
    features = np.concatenate((features_board, features_castling)).astype(np.int8)
    assert features.shape[-1] == 772
    if get_w_persp_result:
        return scipy.sparse.csr_matrix(features), result, wp_res
    return scipy.sparse.csr_matrix(features), result


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


def get_pos_eval(model, fen):
    training = model.training
    model.eval()
    with torch.no_grad():
        feat, res = get_features(fen, "1-0", cond_h_flip=False)
        dense = np.squeeze(np.asarray(feat.todense()))
        t = torch.Tensor(dense).view(1, 772)
        pred = model(t)[0]
    model.train(training)
    return pred, 0.5 + (pred[0] - pred[2]) / 2


#def get_pos_semi_eval(model, fen):
#    training = model.training
#    model.eval()
#    with torch.no_grad():
#        feat, res = get_features(fen, "1-0", cond_h_flip=False)
#        dense = np.squeeze(np.asarray(feat.todense()))
#        t = torch.Tensor(dense).view(1, 772)
#        pred = model.f(t)[0][:, [0,7], 4].view(12, 16, 2)[[5,11]]
#    model.train(training)
#    return pred