import chess
import chess.pgn
import numpy as np
import scipy
import torch
import count

from utils import entropy, string_to_result_class
from chess_utils import get_features, get_pos_eval
from game import ItGame
from loader import CSRDataset, merge_desk


def load_next_game(pgn, print_headers=False):
    """Return ItGame of next game in pgn"""
    g = chess.pgn.read_game(pgn)
    if print_headers:
        print(g.headers)
    return ItGame(g)


def get_max_quiet_entropy(model, game):
    game = ItGame(game)
    game.to_start()
    max_entropy = 0
    max_entropy_pos = None
    while not game.is_final_position():
        game.make_move()
        if game.is_quiet():
            fen = game.fen()
            eval = get_pos_eval(model, fen)[0]
            pos_entropy = entropy(eval)
            if pos_entropy > max_entropy:
                max_entropy = pos_entropy
                max_entropy_pos = fen
    return max_entropy, max_entropy_pos


def get_max_quiet_entropy_pgn(model, pgn_filename):
    # print(f"Loading {pgn}")
    pgn = open(pgn_filename, "r")
    count = 1000
    min_ent = 10
    max_ent = 0
    while count > 0:
        game = chess.pgn.read_game(pgn)
        entropy, fen = get_max_quiet_entropy(model, game)
        if entropy > max_ent:
            max_ent = entropy
            print(fen)
            print(entropy)
            print(get_pos_eval(model, fen)[0])
        if entropy < min_ent:
            min_ent = entropy
            print(fen)
            print(entropy)
            print(get_pos_eval(model, fen)[0])
        count -= 1
    # print(f"Loaded {pgn}")
    return entropy, fen


def extract_fens_from_game(g):
    """Extract fens from a game for training. Returns tuple of fens and final result."""
    game = ItGame(g)
    game.to_start()
    lst = []
    cnt = 1 + np.random.randint(2)
    while not game.is_final_position():
        game.make_move()
        cnt = cnt - 1
        if cnt == 0:
            if not game.is_quiet():
                cnt = 2
            else:
                lst.append(game.fen())
                cnt = 3
    return lst, g.headers["Result"]


def load_all_games(pgn_filename, f=None):
    """Extract games from pgn and optionally perform argument function on each game individually."""
    print(f"Loading gams from {pgn_filename}")
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
            print(f"Loaded {len(lst)} games. Changed results: {count.total_changed}")
    pgn.close()
    print(f"Finished loading {len(lst)} games. Changed results: {count.total_changed}")
    return lst


def data_from_fen_res_set(fens, res):
    features = []
    results = []
    res = string_to_result_class(res)
    og_res = res
    for fen in reversed(fens):
        f, r, res = get_features(fen, res, cond_h_flip=np.random.randint(2), cond_v_flip=np.random.randint(2),
                                 get_w_persp_result=True)
        if res != og_res:
            count.total_changed += 1
        features.append(f)
        results.append(r)
    if len(features) == 0:
        return None
    return scipy.sparse.vstack(features), np.concatenate(results)


# Likely broken now
#def data_from_game_set(game_set):
#    features = []
#    results = []
#    for fen_res_set in game_set:
#        f, r = data_from_fen_res_set(fen_res_set)
#        if f is None:
#            continue
#        features.append(f)
#        results.append(r)
#    return torch.cat(features), torch.cat(results)


def extract_data_from_game(g):
    # Training data is DFRC and must not contain the regular chess starting position.
    # Some older/fan-collected games may still begin from it, so drop any such game.
    # board_fen() is the piece placement only; STARTING_BOARD_FEN matches when both
    # back ranks are the standard rnbqkbnr/RNBQKBNR, i.e. the regular start.
    if g.board().board_fen() == chess.STARTING_BOARD_FEN:
        count.skipped_startpos += 1
        return None
    fens, res = extract_fens_from_game(g)
    return data_from_fen_res_set(fens, res)


def gen_dataset_from_pgn(path="./../pgns/CCRL-404FRCv2.pgn"):
    print(f"generating dataset from {path}")
    f, r = zip(*(load_all_games(path, extract_data_from_game)))
    return scipy.sparse.vstack(f), np.concatenate(r)


def gen_dataset_helper(name, batch_size=16, shuffle=True, save=False, pgn_dir="./../pgns/",
                       out_dir="./../datasets/"):
    print(f"generating dataset from {name}")
    features, results = gen_dataset_from_pgn(f"{pgn_dir}{name}.pgn")
    if save:
        scipy.sparse.save_npz(f"{out_dir}features_{name}.npz", features)
        # Labels are only {0, 1, 2}; int8 stores them exactly at 1/8th the int64 size.
        # The loader casts to torch.long per batch, so the on-disk dtype is irrelevant.
        np.savez(f"{out_dir}targets_{name}.npz", results.astype(np.int8))
    return torch.utils.data.DataLoader(CSRDataset(features, results), batch_size=batch_size, shuffle=shuffle)


def gen_subset_dataset(features, results, n, m=1):
    idx = torch.randint(n, (features.shape[0],)) >= m
    return features[idx], results[idx]


def load_desk(num, generate, save_dir="./datasets/"):
    if generate:
        fd, rd = gen_dataset_from_pgn(f"./../pgns/dfrc-self-play-v{num}.pgn")
        scipy.sparse.save_npz(f"{save_dir}features_desk_v{num}.npz", fd)
        np.savez(f"{save_dir}targets_desk_v{num}.npz", rd)
    else:
        fd = scipy.sparse.load_npz(f"{save_dir}features_desk_v{num}.npz")
        rd = np.load(f"{save_dir}targets_desk_v{num}.npz")['arr_0']
    return fd, rd


def load_and_merge_desk(num, generate, old_tag, new_tag):
    fd, rd = load_desk(num, generate)
    return merge_desk(fd, rd, old_tag, new_tag)
