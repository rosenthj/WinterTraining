import argparse
import chess
import numpy as np
import time
import torch
import torch.nn as nn
from scipy.sparse import vstack, save_npz

from model import NetRelH
import utils
from chess_utils import get_board_tensor, tb_probe_result, get_board_from_tensor, tb_res_to_wdl


def add_piece_randomly(board, piece_char):
    pm = {"k": chess.KING, "p": chess.PAWN, "n": chess.KNIGHT, "b": chess.BISHOP, "r": chess.ROOK, "q": chess.QUEEN}
    color = chess.BLACK if piece_char.islower() else chess.WHITE
    piecetype = pm[piece_char.lower()]
    for square in torch.randperm(64):
        if board.piece_at(square) is not None:
            continue
        if piecetype == chess.PAWN and not (7 < square < 56):
            continue
        board.set_piece_at(square, chess.Piece(piecetype, color))
        return


def gen_ending(pieces):
    board = chess.Board()
    while True:
        board.clear()
        if "k" not in pieces:
            add_piece_randomly(board, "k")
        if "K" not in pieces:
            add_piece_randomly(board, "K")
        for piece in pieces:
            add_piece_randomly(board, piece)
        if board.is_check():
            continue
        board.turn = not board.turn
        if board.is_check():
            continue
        board.turn = not board.turn
        return board


def sample_endings(pieces, count, model=None, tablebase=None):
    endings = list()
    while len(endings) < count:
        ending = gen_ending(pieces)
        outcome = tb_probe_result(ending) if tablebase is None else tb_res_to_wdl(tablebase.probe_wdl(ending))
        end_tens = get_board_tensor(ending, sparse=True)
        if model:
            with torch.no_grad():
                eval = model(torch.Tensor(end_tens.todense()).view(1, -1))
                endings.append((end_tens, outcome, eval))
        else:
            endings.append((end_tens, outcome))
    return endings


def gen_ending_dataset(pieces_count_list, tablebase=None):
    features = list()
    labels = list()
    for pieces, count in pieces_count_list:
        print(f"Generating {count} endings for {pieces}")
        endings = sample_endings(pieces, count, tablebase=tablebase)
        features.append(vstack([ending[0] for ending in endings]))
        labels.append(np.array([ending[1] for ending in endings]))
    features = vstack(features)
    labels = np.concatenate(labels)
    return features, labels


def main():

    # Command line argument parsing
    parser = argparse.ArgumentParser(description="Script to transform pgns into datasets used for training")
    parser.add_argument('--pieces', type=str, default="RrPp")
    args = parser.parse_args()

    # model = NetRelH(d=16, fd=64, num_inputs=768, activation=nn.Hardtanh(min_val=0, max_val=8))
    # utils.load_weights(model, "rnet16H64l", 7)
    # endings = sample_endings(args.pieces, 100)
    count = 48000
    piece_count_list = list()
    w_pieces = ["P", "N", "B", "R", "Q"]
    b_pieces = [piece.lower() for piece in w_pieces]
    for w_piece in w_pieces:
        for b_piece in b_pieces:
            for pawn_config in ["P", "p", "PP", "pp", "Pp"]:
                piece_count_list.append(((w_piece + b_piece + pawn_config), count))
    piece_count_list.append(("P", count))
    piece_count_list.append(("p", count))
    piece_count_list.append(("BP", count))
    piece_count_list.append(("bp", count))
    piece_count_list.append(("RBr", count))
    piece_count_list.append(("Rbr", count))
    piece_count_list.append(("nP", count // 10))
    piece_count_list.append(("Np", count // 10))
    piece_count_list.append(("bP", count // 10))
    piece_count_list.append(("Bp", count // 10))
    start_time = time.time()
    with chess.syzygy.open_tablebase("../../../Chess/TB_Merged") as tablebase:
        features, labels = gen_ending_dataset(piece_count_list, tablebase=tablebase)
    run_time = time.time() - start_time
    if run_time < 600:
        print(f"Runtime: {run_time} seconds")
    elif run_time < 3600:
        print(f"Runtime: {run_time // 60}:{run_time % 60}")
    else:
        print(f"Runtime: {run_time // 3600}:{(run_time % 3600) // 60}:{run_time % 60}")
    print(features.shape)
    print(labels.shape)
    save_npz(f"./../features_desk_vEnd.npz", features)
    np.savez(f"./../targets_desk_vEnd.npz", labels)

    return 0


if __name__ == '__main__':
    main()
