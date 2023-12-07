import chess
import numpy as np
import scipy
import torch

from game import ItGame
from loader import CSRDataset, merge_desk


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
        if (len(lst) % 25000) == 0:
            print("Loaded {} games".format(len(lst)))
    pgn.close()
    print("Finished loading {} games".format(len(lst)))
    return lst


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


def gen_dataset_helper(name, batch_size=16, shuffle=True, save=False):
    features, results = gen_dataset_from_pgn(f"./../Data/{name}.pgn")
    if save:
        scipy.sparse.save_npz(f"{name}_features.npz", features)
        np.savez(f"{name}_targets.npz", results)
    return torch.utils.data.DataLoader(CSRDataset(features,results), batch_size=batch_size, shuffle=shuffle)


def gen_subset_dataset(features, results, n, m=1):
    idx = torch.randint(n, (features.shape[0],)) >= m
    return features[idx], results[idx]


def load_desk(num, generate, save_dir="./"):
    if generate:
        fd, rd = gen_dataset_from_pgn(f"./../Data/dfrc-self-play-v{num}.pgn")
        scipy.sparse.save_npz(f"{save_dir}features_desk_v{num}.npz", fd)
        np.savez(f"{save_dir}targets_desk_v{num}.npz", rd)
    else:
        fd = scipy.sparse.load_npz(f"{save_dir}features_desk_v{num}.npz")
        rd = np.load(f"{save_dir}targets_desk_v{num}.npz")['arr_0']
    return fd, rd


def load_and_merge_desk(num, generate, old_tag, new_tag):
    fd, rd = load_desk(num, generate)
    return merge_desk(fd, rd, old_tag, new_tag)