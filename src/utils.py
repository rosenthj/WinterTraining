import chess
import numpy as np
import scipy
import torch


def string_to_result_class(result_str):
    if result_str == '1-0':
        return 0
    elif result_str == '1/2-1/2':
        return 1
    elif result_str == '0-1':
        return 2
    raise ValueError(f"Unexpected value: {result_str}")


def flip_result(result):
    return 2 - result


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