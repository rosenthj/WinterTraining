import argparse
import glob
import os
import time

import chess
import chess.syzygy
import numpy as np
import scipy.sparse

import config
import count
from chess_utils import get_features


DEFAULT_FENS_ROOT = "/home/jonathan/Documents/Research/Chess/GFN/Gilbreth/fens_v2"


def discover_csv_files(fens_root, engine_glob="Winter*", method_suffix="_novel_mutate"):
    """Find all fens_*.csv files under <root>/<engine>*/<*method_suffix>/, sorted."""
    pattern = os.path.join(fens_root, engine_glob, f"*{method_suffix}", "fens_*.csv")
    return sorted(glob.glob(pattern))


def dedup_key(board):
    """Position key from the side-to-move perspective (clocks dropped)."""
    if board.turn == chess.BLACK:
        board = board.mirror()
    return board.epd()


def iter_row_positions(fen, move_str, stats):
    """Yield the FENs before and after the engine move; skip unusable moves."""
    board = chess.Board(chess960=True)
    try:
        board.set_fen(fen)
    except ValueError:
        stats["bad_fen"] += 1
        return
    yield fen, board.copy(stack=False)
    if move_str == "illegal":
        stats["illegal_move"] += 1
        return
    try:
        move = chess.Move.from_uci(move_str)
    except ValueError:
        stats["bad_move"] += 1
        return
    if move not in board.legal_moves:
        stats["bad_move"] += 1
        return
    board.push(move)
    yield board.fen(), board


def process_files(csv_files, limit=None, skip_checks=False):
    seen = set()
    file_feature_blocks = []
    labels = []
    stats = {
        "rows": 0,
        "bad_line": 0,
        "bad_fen": 0,
        "illegal_move": 0,
        "bad_move": 0,
        "duplicates": 0,
        "in_check": 0,
        "probe_failed": 0,
        "positions": 0,
    }
    start_time = time.time()
    done = False
    for file_idx, path in enumerate(csv_files):
        rows = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 2 or not parts[0] or not parts[1]:
                    stats["bad_line"] += 1
                    continue
                stats["rows"] += 1
                for pos_fen, board in iter_row_positions(parts[0], parts[1], stats):
                    if skip_checks and board.is_check():
                        stats["in_check"] += 1
                        continue
                    key = dedup_key(board)
                    if key in seen:
                        stats["duplicates"] += 1
                        continue
                    seen.add(key)
                    # Dummy result string: every position here has <=6 pieces and no
                    # castling rights, so get_features overwrites it with the exact
                    # tablebase WDL from the side-to-move perspective.
                    try:
                        features, result = get_features(pos_fen, "1/2-1/2",
                                                        cond_h_flip=False, cond_v_flip=False)
                    except (KeyError, chess.syzygy.MissingTableError) as e:
                        seen.discard(key)
                        stats["probe_failed"] += 1
                        print(f"Probe failed for '{pos_fen}': {e}")
                        continue
                    rows.append(features)
                    labels.append(int(result[0]))
                    stats["positions"] += 1
                if limit is not None and stats["rows"] >= limit:
                    done = True
                    break
        if rows:
            file_feature_blocks.append(scipy.sparse.vstack(rows))
        if done or (file_idx + 1) % 100 == 0:
            elapsed = time.time() - start_time
            print(f"[{file_idx + 1}/{len(csv_files)}] {os.path.basename(os.path.dirname(path))}: "
                  f"{stats['rows']} rows, {stats['positions']} positions, "
                  f"{stats['duplicates']} duplicates, {elapsed:.0f}s")
        if done:
            break
    if not file_feature_blocks:
        raise RuntimeError("No positions extracted")
    features = scipy.sparse.vstack(file_feature_blocks)
    labels = np.array(labels, dtype=np.int8)
    return features, labels, stats


def main():
    parser = argparse.ArgumentParser(
        description="Build a dataset from novel-search engine-mistake FEN csvs "
                    "(positions before and after the engine move, deduplicated)")
    parser.add_argument('--fens-root', type=str, default=DEFAULT_FENS_ROOT,
                        help="Root directory containing the <Engine>_us__move_* run directories")
    parser.add_argument('--engine', type=str, default="Winter",
                        help="Engine name prefix of run directories to include")
    parser.add_argument('--method', type=str, default="novel_mutate",
                        help="Generation-method subdirectory suffix to include")
    parser.add_argument('--name', type=str, default="desk_vNovel",
                        help="Base name of the generated dataset files")
    parser.add_argument('--out-dir', type=str, default="./../datasets/",
                        help="Directory to write features_{name}.npz and targets_{name}.npz")
    parser.add_argument('--tablebase', type=str, default="../../../Chess/TB_Merged",
                        help="Path to the Syzygy tablebase directory used to label positions")
    parser.add_argument('--limit', type=int, default=None,
                        help="Stop after this many csv rows (for smoke testing)")
    parser.add_argument('--skip-checks', action='store_true',
                        help="Exclude positions where the side to move is in check "
                             "(static evaluation is never called in check)")
    args = parser.parse_args()

    csv_files = discover_csv_files(args.fens_root, engine_glob=f"{args.engine}*",
                                   method_suffix=f"_{args.method}")
    dirs = sorted({os.path.basename(os.path.dirname(p)) for p in csv_files})
    print(f"Found {len(csv_files)} csv files in {len(dirs)} directories:")
    for d in dirs:
        print(f"  {d}")

    config.tablebase = chess.syzygy.open_tablebase(args.tablebase)
    features, labels, stats = process_files(csv_files, limit=args.limit,
                                            skip_checks=args.skip_checks)
    config.tablebase.close()

    print(f"Stats: {stats}")
    print(f"TB Queries: {count.total_tb_queries}")
    print(f"Features: {features.shape}, targets: {labels.shape}, "
          f"label counts: {np.bincount(labels, minlength=3)}")
    scipy.sparse.save_npz(os.path.join(args.out_dir, f"features_{args.name}.npz"), features)
    np.savez(os.path.join(args.out_dir, f"targets_{args.name}.npz"), labels)

    return 0


if __name__ == '__main__':
    main()
