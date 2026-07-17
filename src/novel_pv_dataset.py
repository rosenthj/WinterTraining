"""PV-neighbourhood dataset from novel-search adversarial positions.

The adversarial positions themselves are often evaluated fine by the network --
the *search* was fooled because positions downstream were misevaluated. So
instead of random perturbations (see novel_balanced_dataset.py), harvest the
positions the search actually relies on. For each adversarial position:

  1. the position itself;
  2. the position after every legal move;
  3. every position along the engine's PV, searched at the same node count the
     adversarial position was generated with (read from the directory name);
  4. for every legal move that preserves the EGTB-best outcome of the position
     (win/draw/loss class, cursed = draw) and is not the engine's move, every
     position along the engine's PV after playing that move (same node count).

Only directories at or below --max-nodes are used (high-node searches are too
slow to reproduce, and those directories hold few samples anyway). Base
positions are processed rare-material-first exactly as in the balanced
dataset, so stopping at --target-positions keeps rare signatures complete.
Positions where the side to move is in check and terminal positions are
excluded; every stored position is labeled with its exact Syzygy WDL from the
side-to-move perspective, deduplicated globally, no flip augmentation.
"""
import argparse
import os
import re
import time

import chess
import chess.engine
import chess.syzygy
import numpy as np
import scipy.sparse

import config
import count
from chess_utils import get_features, tb_res_to_wdl
from novel_fens_to_dataset import DEFAULT_FENS_ROOT, discover_csv_files, dedup_key
from novel_balanced_dataset import material_signature, usable, rss_gb, balanced_order

_NODES_RE = re.compile(r'_n(\d+)_')


def nodes_from_path(path):
    """Node count encoded in the run directory name, e.g. Winter_us__move_n800_6."""
    m = _NODES_RE.search(path)
    if m is None:
        raise ValueError(f"No node count in path: {path}")
    return int(m.group(1))


def scan_base_positions(csv_files, max_nodes, limit=None):
    """Unique adversarial positions with engine move, node count and signature."""
    seen = set()
    fens, moves, node_counts, sig_codes = [], [], [], []
    sig_ids = {}
    sig_names = []
    rows = 0
    start = time.time()
    done = False
    for file_idx, path in enumerate(csv_files):
        nodes = nodes_from_path(path)
        if nodes > max_nodes:
            continue
        with open(path, "r") as f:
            for line in f:
                parts = [p.strip() for p in line.strip().split(",")]
                if len(parts) < 2 or not parts[0] or not parts[1]:
                    continue
                rows += 1
                board = chess.Board(chess960=True)
                try:
                    board.set_fen(parts[0])
                except ValueError:
                    continue
                key = dedup_key(board)
                if key in seen:
                    continue
                seen.add(key)
                fens.append(parts[0])
                moves.append(parts[1])
                node_counts.append(nodes)
                sig = material_signature(board)
                code = sig_ids.get(sig)
                if code is None:
                    code = sig_ids[sig] = len(sig_names)
                    sig_names.append(sig)
                sig_codes.append(code)
                if limit is not None and rows >= limit:
                    done = True
                    break
        if done:
            break
        if (file_idx + 1) % 500 == 0:
            print(f"scan [{file_idx + 1}/{len(csv_files)}]: {rows} rows, "
                  f"{len(fens)} unique base positions, {time.time() - start:.0f}s, "
                  f"rss {rss_gb():.1f}GB", flush=True)
    return (fens, moves, np.asarray(node_counts, dtype=np.int32),
            np.asarray(sig_codes, dtype=np.int32), sig_names, rows)


def pv_positions(engine, board, nodes, game):
    """Positions along the engine's PV from ``board`` (excluding ``board``)."""
    info = engine.analyse(board, chess.engine.Limit(nodes=nodes), game=game)
    out = []
    b = board.copy(stack=False)
    for move in info.get("pv", []):
        b.push(move)
        out.append(b.copy(stack=False))
    return out


def expand_base(fen, move_str, nodes, engine, game, stats):
    """Candidate boards for one adversarial position (may include unusable ones)."""
    board = chess.Board(chess960=True)
    board.set_fen(fen)
    try:
        engine_move = chess.Move.from_uci(move_str)
    except ValueError:
        engine_move = None

    candidates = [board]
    children = []  # (move, child, mover WDL class 0/1/2)
    for move in board.legal_moves:
        b = board.copy(stack=False)
        b.push(move)
        candidates.append(b)
        try:
            wdl_class = tb_res_to_wdl(-config.tablebase.probe_wdl(b))
        except (KeyError, chess.syzygy.MissingTableError):
            stats["rank_probe_failed"] += 1
            wdl_class = None
        children.append((move, b, wdl_class))

    stats["searches"] += 1
    candidates.extend(pv_positions(engine, board, nodes, game))

    classes = [c for _, _, c in children if c is not None]
    if classes:
        best_class = min(classes)
        for move, child, wdl_class in children:
            # Never search terminal children: with no legal move Winter answers
            # "bestmove a1a1", which python-chess rejects, breaking the protocol.
            if wdl_class == best_class and move != engine_move and not child.is_game_over():
                stats["searches"] += 1
                stats["alt_pvs"] += 1
                candidates.extend(pv_positions(engine, child, nodes, game))
    return candidates


def generate(fens, moves, node_counts, rank, order, target, engine_bin):
    engine = chess.engine.SimpleEngine.popen_uci(engine_bin)
    seen = set()
    blocks, rows = [], []
    label_blocks, labels = [], []
    stats = {"base_used": 0, "max_rank": 0, "searches": 0, "alt_pvs": 0,
             "candidates": 0, "unusable": 0, "duplicates": 0,
             "rank_probe_failed": 0, "probe_failed": 0, "engine_errors": 0,
             "positions": 0}
    start = time.time()
    try:
        for idx in order:
            if stats["positions"] >= target:
                break
            stats["base_used"] += 1
            stats["max_rank"] = max(stats["max_rank"], int(rank[idx]) + 1)
            # A fresh game per base clears the hash table, so every PV comes from
            # a cold search like the one that produced the adversarial sample.
            try:
                candidates = expand_base(fens[idx], moves[idx], int(node_counts[idx]),
                                         engine, stats["base_used"], stats)
            except (chess.engine.EngineError, chess.engine.EngineTerminatedError) as e:
                stats["engine_errors"] += 1
                print(f"Engine error on '{fens[idx]}': {e}; restarting engine", flush=True)
                try:
                    engine.quit()
                except Exception:
                    engine.close()
                engine = chess.engine.SimpleEngine.popen_uci(engine_bin)
                continue
            for board in candidates:
                stats["candidates"] += 1
                if not usable(board):
                    stats["unusable"] += 1
                    continue
                key = dedup_key(board)
                if key in seen:
                    stats["duplicates"] += 1
                    continue
                seen.add(key)
                try:
                    features, result = get_features(board.fen(), "1/2-1/2",
                                                    cond_h_flip=False, cond_v_flip=False)
                except (KeyError, chess.syzygy.MissingTableError) as e:
                    seen.discard(key)
                    stats["probe_failed"] += 1
                    print(f"Probe failed for '{board.fen()}': {e}")
                    continue
                rows.append(features)
                labels.append(int(result[0]))
                stats["positions"] += 1
            if len(rows) >= 200_000:
                blocks.append(scipy.sparse.vstack(rows))
                label_blocks.append(np.asarray(labels, dtype=np.int8))
                rows, labels = [], []
            if stats["base_used"] % 10_000 == 0:
                print(f"gen: {stats['base_used']} base positions, {stats['positions']} kept, "
                      f"{stats['searches']} searches, {stats['duplicates']} duplicates, "
                      f"{time.time() - start:.0f}s, rss {rss_gb():.1f}GB", flush=True)
    finally:
        engine.quit()
    if rows:
        blocks.append(scipy.sparse.vstack(rows))
        label_blocks.append(np.asarray(labels, dtype=np.int8))
    if not blocks:
        raise RuntimeError("No positions generated")
    return scipy.sparse.vstack(blocks), np.concatenate(label_blocks), stats


def main():
    parser = argparse.ArgumentParser(
        description="PV-neighbourhood dataset from novel-search adversarial positions")
    parser.add_argument('--fens-root', type=str, default=DEFAULT_FENS_ROOT)
    parser.add_argument('--engine', type=str, default="Winter")
    parser.add_argument('--method', type=str, default="novel_mutate")
    parser.add_argument('--engine-bin', type=str, default="./../Winter",
                        help="UCI engine used to reproduce the PVs")
    parser.add_argument('--name', type=str, default="desk_vNovelPv")
    parser.add_argument('--out-dir', type=str, default="./../datasets_aux/")
    parser.add_argument('--tablebase', type=str, default="../../../Chess/TB_Merged")
    parser.add_argument('--max-nodes', type=int, default=12800,
                        help="Skip run directories generated with more nodes than this")
    parser.add_argument('--target-positions', type=int, default=10_000_000,
                        help="Stop once this many positions are stored")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--limit', type=int, default=None,
                        help="Stop scanning after this many csv rows (smoke testing)")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    csv_files = discover_csv_files(args.fens_root, engine_glob=f"{args.engine}*",
                                   method_suffix=f"_{args.method}")
    print(f"Found {len(csv_files)} csv files")

    fens, moves, node_counts, sig_codes, sig_names, rows = scan_base_positions(
        csv_files, args.max_nodes, limit=args.limit)
    print(f"Scanned {rows} rows (dirs <= n{args.max_nodes}): "
          f"{len(fens)} unique adversarial positions")
    rank, order = balanced_order(sig_codes, sig_names, rng)
    del sig_codes

    config.tablebase = chess.syzygy.open_tablebase(args.tablebase)
    features, labels, stats = generate(fens, moves, node_counts, rank, order,
                                       args.target_positions, args.engine_bin)
    config.tablebase.close()

    print(f"Stats: {stats}")
    print(f"Per-signature cap reached: {stats['max_rank']} "
          f"(base positions used: {stats['base_used']}/{len(fens)})")
    print(f"TB Queries: {count.total_tb_queries}")
    print(f"Features: {features.shape}, targets: {labels.shape}, "
          f"label counts: {np.bincount(labels, minlength=3)}")
    scipy.sparse.save_npz(os.path.join(args.out_dir, f"features_{args.name}.npz"), features)
    np.savez(os.path.join(args.out_dir, f"targets_{args.name}.npz"), labels)

    return 0


if __name__ == '__main__':
    main()
