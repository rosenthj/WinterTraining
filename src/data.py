import chess
import numpy as np
import scipy
import torch

from game import ItGame
from utils import get_standardised_board_and_result


def load_next_game(pgn, print_headers=False):
    """Return ItGame of next game in pgn"""
    g = chess.pgn.read_game(pgn)
    if print_headers:
        print(g.headers)
    return ItGame(g)


def extract_fens_from_game(g):
    """Extract fens from a game for training. Returns tupple of fens and final result."""
    game = ItGame(g)
    game.to_start()
    lst = []
    # assert (game._umake != None)
    # while (game.castling_rights() != 0 or not game.is_quiet()) and not game.is_final_position():
    #    game.make_move()
    # if game.is_final_position():
    #    return None
    # lst.append(game.fen())
    hmvc_dec = 0
    cnt = 1 + np.random.randint(2)
    while not game.is_final_position():
        game.make_move()
        if game.halfmove_clock() == 0:
            hmvc_dec = 0
        cnt = cnt - 1
        if cnt == 0:
            if hmvc_dec > 0 or not game.is_quiet():
                cnt = 2
                hmvc_dec -= 1
            else:
                lst.append(game.fen())
                hmvc_dec = 10
                cnt = 5
    return lst, g.headers["Result"]


def load_all_games(pgn_filename, f=None):
    """Extract games from pgn and optionally perform argument function on each game individually."""
    lst = []
    pgn = open(pgn_filename, "r")
    # g = load_next_game(pgn)
    # g.print_board
    g = chess.pgn.read_game(pgn)
    while (g):
        if f:
            g = f(g)
        if g:
            lst.append(g)
        # g = load_next_game(pgn)
        g = chess.pgn.read_game(pgn)
        if (len(lst) % 5000) == 0:  # or (len(lst) > 15700 and len(lst) < 15800):
            print("Loaded {} games".format(len(lst)))
            if len(lst) >= 365000:
                break
    pgn.close()
    print("Finished loading {} games".format(len(lst)))
    return lst


def add_tag(pgn_in: str = "./../Data/CCRL-404FRC.pgn", pgn_out: str = "./../Data/CCRL-404FRCv2.pgn"):
    """Add pgn tag to input pgn and store to output pgn"""
    pgn = open(pgn_in, "r")
    pgno = open(pgn_out, "w")
    for line in pgn:
        if "FEN" in line and "-\"" in line:
            lp = line.split("\"")
            line = "{}\"{} 0 1\"{}".format(lp[0], lp[1], lp[2])
        if "Variant" not in line:
            pgno.write(line)
        if "Result" in line:
            pgno.write("[Variant \"chess 960\"]\n")


def get_features(fen, result_str, cond_h_flip=False, cond_v_flip=False):
    board, result = get_standardised_board_and_result(fen, result_str, cond_h_flip, cond_v_flip)
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
    return scipy.sparse.csr_matrix(features), result


def data_from_fen_res_set(fen_res_tuple):
    features = []
    results = []
    for idx in range(len(fen_res_tuple[0])):
        f, r = get_features(fen_res_tuple[0][idx], fen_res_tuple[1], np.random.randint(2), np.random.randint(2))
        features.append(f)
        results.append(r)
    if len(features) == 0:
        return None
    return scipy.sparse.vstack(features), np.concatenate(results)


# Likely broken now
def data_from_game_set(game_set):
    features = []
    results = []
    for fen_res_set in game_set:
        f, r = data_from_fen_res_set(fen_res_set)
        if f is None:
            continue
        features.append(f)
        results.append(r)
    return torch.cat(features), torch.cat(results)


def extract_data_from_game(g):
    fen_res_tuple = extract_fens_from_game(g)
    if fen_res_tuple is None:
        return None
    return data_from_fen_res_set(fen_res_tuple)


def gen_dataset_from_pgn(path="./../Data/CCRL-404FRCv2.pgn"):
    f, r = zip(*(load_all_games(path, extract_data_from_game)))
    return scipy.sparse.vstack(f), np.concatenate(r)


def load_features_results(name):
    features = scipy.sparse.load_npz(f"{name}_features.npz")
    results = np.load(f"{name}_targets.npz")['arr_0']
    return features, results


def load_dataset(name, batch_size=16, shuffle=True):
    features, results = load_features_results(name)
    return torch.utils.data.DataLoader(CSRDataset(features,results), batch_size=batch_size, shuffle=shuffle)


def gen_dataset_helper(name, batch_size=16, shuffle=True, save=False):
    features, results = gen_dataset_from_pgn(f"./../Data/{name}.pgn")
    if save:
        scipy.sparse.save_npz(f"{name}_features.npz", features)
        np.savez(f"{name}_targets.npz", results)
    return torch.utils.data.DataLoader(CSRDataset(features,results), batch_size=batch_size, shuffle=shuffle)


def gen_subset_dataset(features, results, n, m=1):
    idx = torch.randint(n, (features.shape[0],)) >= m
    return features[idx], results[idx]


class CSRDataset:
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __getitem__(self, index):
        return np.squeeze(np.asarray(self.features[index].todense())), self.targets[index]

    def __len__(self):
        return self.features.shape[0]
