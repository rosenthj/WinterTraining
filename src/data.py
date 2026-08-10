import chess
import chess.pgn
import numpy as np
import scipy
import torch
import config
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


def is_tb_position(fen):
    """True if get_features would rewrite this fen's label from the tablebase.

    Mirrors the condition in chess_utils.get_standardised_board_and_result (at most six
    pieces and no castling rights) but reads it straight off the fen string, since the
    board transforms applied there change neither the piece count nor whether any
    castling rights exist.
    """
    placement, _, castling = fen.split(' ', 3)[:3]
    if castling != '-':
        return False
    return sum(c.isalpha() for c in placement) <= 6


def data_from_fen_res_set(fens, res, force_tb=False):
    features = []
    results = []
    res = string_to_result_class(res)
    og_res = res
    # Piece count never increases, so the tablebase positions form a suffix of the game.
    # Walking backwards each probe overwrites the previous one, which means only the
    # earliest tablebase position -- the entry point into the endgame -- actually
    # propagates its result into the rest of the game; the deeper ones only relabel
    # themselves. So the entry point is always probed, and the deeper positions are
    # relabelled only with probability tb_relabel_prob. The rest keep the game's actual
    # outcome, which retains the practical difficulty of converting the endgame (and
    # skips the probe).
    p = 1.0 if force_tb else config.tb_relabel_prob
    entry = -1
    if p < 1.0:
        for i, fen in enumerate(fens):
            if is_tb_position(fen):
                entry = i
                break
    for i in range(len(fens) - 1, -1, -1):
        fen = fens[i]
        probe = True
        if p < 1.0 and i != entry and is_tb_position(fen):
            probe = np.random.random() < p
        # A skipped position must be labelled from the game result rather than the
        # threaded res, otherwise a deeper probe would leak into it.
        f, r, probed_res = get_features(fen, res if probe else og_res,
                                        cond_h_flip=np.random.randint(2), cond_v_flip=np.random.randint(2),
                                        get_w_persp_result=True, tb_probe=probe)
        if probe:
            res = probed_res
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


def abnormal_final_board(g):
    """Return the final board if the game's decisive result is unexplained by any chess rule.

    A time forfeit is recorded by writing the loser's final move and then scoring the game
    against them, so the game stops at a position that is neither mate, stalemate,
    insufficient material, a repetition, nor a 50-move draw. The recorded result is then
    unrelated to the position: across desk_v312 the forfeiting side was materially ahead
    almost exactly as often as behind, and 116 games ended with the result inverted
    relative to the Syzygy verdict.

    The board is returned rather than a bool because the caller needs its piece count to
    decide whether the game is salvageable -- see extract_data_from_game.

    Repetition is tested at twofold, the permissive reading, so that a genuine repetition
    is never mistaken for a forfeit.
    """
    if g.headers["Result"] == "1/2-1/2":
        return None
    board = g.end().board()
    if (board.is_checkmate() or board.is_stalemate() or board.is_insufficient_material()
            or board.is_repetition(2) or board.halfmove_clock >= 100):
        return None
    return board


def extract_data_from_game(g):
    count.total_games += 1
    # Training data is DFRC and must not contain the regular chess starting position.
    # Some older/fan-collected games may still begin from it, so drop any such game.
    # board_fen() is the piece placement only; STARTING_BOARD_FEN matches when both
    # back ranks are the standard rnbqkbnr/RNBQKBNR, i.e. the regular start.
    if g.board().board_fen() == chess.STARTING_BOARD_FEN:
        count.skipped_startpos += 1
        return None
    # A time forfeit's recorded result is bogus, but it is only ever used as the seed of the
    # backwards walk in data_from_fen_res_set: if the game reached a tablebase position, the
    # probe at the entry into the endgame overwrites it and every earlier position is labelled
    # from the Syzygy verdict instead. Such a game is therefore fully repairable and is kept.
    # Without a tablebase position there is nothing to override the result, so it is dropped.
    forfeit_board = abnormal_final_board(g) if config.drop_abnormal else None
    if forfeit_board is not None and len(forfeit_board.piece_map()) > 6:
        # Piece count never increases, so a game ending above six pieces never held a tablebase
        # position. Checked before extracting fens, which is much the more expensive half.
        count.skipped_abnormal += 1
        return None
    fens, res = extract_fens_from_game(g)
    if forfeit_board is not None and not any(is_tb_position(f) for f in fens):
        # Reached six pieces, but no position that survived sampling is probeable.
        count.skipped_abnormal += 1
        return None
    if forfeit_board is not None:
        count.repaired_abnormal += 1
    # The repair is the only reason a forfeited game is usable, so it must not be subject to
    # tb_relabel_prob -- those games always take every probe.
    return data_from_fen_res_set(fens, res, force_tb=forfeit_board is not None)


def gen_dataset_from_pgn(path="./../pgns/CCRL-404FRCv2.pgn"):
    print(f"generating dataset from {path}")
    f, r = zip(*(load_all_games(path, extract_data_from_game)))
    return scipy.sparse.vstack(f), np.concatenate(r)


def gen_dataset_helper(name, batch_size=16, shuffle=True, save=False, pgn_dir="./../pgns/",
                       out_dir="./../datasets/", out_name=None):
    """Build a dataset from {pgn_dir}{name}.pgn, writing it under ``out_name`` if given.

    Separating the two lets a regenerated dataset carry a revision tag (desk_v311 ->
    desk_v311a) without renaming the PGN. loader.newest_variants then prefers the
    revision over the original for the same version number.
    """
    print(f"generating dataset from {name}")
    features, results = gen_dataset_from_pgn(f"{pgn_dir}{name}.pgn")
    out_name = out_name or name
    if save:
        scipy.sparse.save_npz(f"{out_dir}features_{out_name}.npz", features)
        # Labels are only {0, 1, 2}; int8 stores them exactly at 1/8th the int64 size.
        # The loader casts to torch.long per batch, so the on-disk dtype is irrelevant.
        np.savez(f"{out_dir}targets_{out_name}.npz", results.astype(np.int8))
        print(f"wrote {out_dir}features_{out_name}.npz and {out_dir}targets_{out_name}.npz")
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
