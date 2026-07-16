"""Material-balanced dataset from novel-search adversarial positions.

The plain desk_vNovel dataset over-represents whatever the novel-search mutation
chains clustered on. This generator rebalances by material signature and widens
each adversarial position into its local neighbourhood:

  1. the adversarial position itself;
  2. the position after every legal move in it;
  3. one random perturbation of it (one piece moved to another empty square);
  4. one random perturbation of the position after the engine's move.

Base positions are processed rare-material-first (round-robin over signatures),
so stopping at --target-positions keeps every rare signature complete and
truncates only the common ones. Positions where the side to move is in check
and terminal positions (checkmate/stalemate) are excluded; every stored
position is labeled with its exact Syzygy WDL from the side-to-move
perspective, deduplicated globally, no h/v flip augmentation.
"""
import argparse
import os
import time

import chess
import chess.syzygy
import numpy as np
import scipy.sparse

import config
import count
from chess_utils import get_features
from novel_fens_to_dataset import DEFAULT_FENS_ROOT, discover_csv_files, dedup_key

PIECE_ORDER = {s: i for i, s in enumerate("KQRBNP")}


def rss_gb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS"):
                return int(line.split()[1]) / 1e6
    return 0.0


def material_signature(board):
    """Material from the side-to-move perspective, e.g. 'KQR v KRP'."""
    b = board if board.turn == chess.WHITE else board.mirror()
    own, opp = [], []
    for piece in b.piece_map().values():
        (own if piece.color == chess.WHITE else opp).append(piece.symbol().upper())
    own.sort(key=PIECE_ORDER.get)
    opp.sort(key=PIECE_ORDER.get)
    return "".join(own) + "v" + "".join(opp)


def usable(board):
    """Positions the static eval can actually be called on."""
    return not board.is_check() and not board.is_stalemate()


def perturb(board, rng, tries=20):
    """Move one random piece to another empty square; None if no valid result.

    Guards: pawns stay off the back ranks, the resulting position must be
    legal (both kings, side not to move not in check, ...) and usable
    (side to move not in check, not terminal).
    """
    from_squares = list(board.piece_map())
    for _ in range(tries):
        b = board.copy(stack=False)
        b.ep_square = None  # move history no longer applies to the mutated position
        from_sq = from_squares[rng.integers(len(from_squares))]
        piece = b.remove_piece_at(from_sq)
        empties = [sq for sq in chess.SQUARES
                   if b.piece_at(sq) is None and sq != from_sq
                   and not (piece.piece_type == chess.PAWN
                            and chess.square_rank(sq) in (0, 7))]
        if not empties:
            continue
        b.set_piece_at(empties[rng.integers(len(empties))], piece)
        if b.is_valid() and usable(b):
            return b
    return None


def scan_base_positions(csv_files, limit=None):
    """Collect the unique adversarial positions with their engine move and signature.

    Signatures are returned as integer codes (plus the code -> name list) and the
    memory-heavy dedup set is dropped on return; with ~13M base positions the
    peak footprint here dominates the whole run, so keep it lean.
    """
    seen = set()
    fens, moves, sig_codes = [], [], []
    sig_ids = {}
    sig_names = []
    rows = 0
    start = time.time()
    done = False
    for file_idx, path in enumerate(csv_files):
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
    return fens, moves, np.asarray(sig_codes, dtype=np.int32), sig_names, rows


def balanced_order(sig_codes, sig_names, rng):
    """Round-robin order over material signatures, as numpy arrays.

    Returns ``(rank, order)``: ``rank[i]`` is base position i's random rank
    within its signature, and ``order`` lists base-position indices sorted by
    rank (random tie-break). Position r of every signature comes before
    position r+1 of any signature, so a prefix of this order is exactly the
    per-signature cap that fits the budget: rare signatures complete, common
    ones truncated. All-numpy to keep ~13M entries at ~150MB instead of the
    multiple GB a list-of-tuples sort would need.
    """
    counts = np.bincount(sig_codes, minlength=len(sig_names))
    print(f"{len(sig_names)} material signatures; most common:")
    for code in np.argsort(-counts)[:15]:
        print(f"  {sig_names[code]}: {counts[code]}")
    sorted_by_sig = np.argsort(sig_codes, kind="stable")
    rank = np.empty(len(sig_codes), dtype=np.int32)
    pos = 0
    for cnt in counts:
        rank[sorted_by_sig[pos:pos + cnt]] = rng.permutation(cnt)
        pos += cnt
    order = np.lexsort((rng.random(len(sig_codes)), rank))
    return rank, order


def expand_base(fen, move_str, rng, perturb_tries):
    """Candidate boards for one adversarial position (may include unusable ones)."""
    board = chess.Board(chess960=True)
    board.set_fen(fen)
    candidates = [board]
    after = None
    try:
        engine_move = chess.Move.from_uci(move_str)
    except ValueError:
        engine_move = None
    for move in board.legal_moves:
        b = board.copy(stack=False)
        b.push(move)
        candidates.append(b)
        if move == engine_move:
            after = b
    p = perturb(board, rng, tries=perturb_tries)
    if p is not None:
        candidates.append(p)
    if after is not None:
        p = perturb(after, rng, tries=perturb_tries)
        if p is not None:
            candidates.append(p)
    return candidates


def generate(fens, moves, rank, order, target, rng, perturb_tries):
    seen = set()
    blocks, rows = [], []
    label_blocks, labels = [], []
    stats = {"base_used": 0, "max_rank": 0, "candidates": 0, "unusable": 0,
             "duplicates": 0, "probe_failed": 0, "positions": 0}
    start = time.time()
    for idx in order:
        if stats["positions"] >= target:
            break
        stats["base_used"] += 1
        stats["max_rank"] = max(stats["max_rank"], int(rank[idx]) + 1)
        for board in expand_base(fens[idx], moves[idx], rng, perturb_tries):
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
        if stats["base_used"] % 50_000 == 0:
            print(f"gen: {stats['base_used']} base positions, {stats['positions']} kept, "
                  f"{stats['duplicates']} duplicates, {time.time() - start:.0f}s, "
                  f"rss {rss_gb():.1f}GB", flush=True)
    if rows:
        blocks.append(scipy.sparse.vstack(rows))
        label_blocks.append(np.asarray(labels, dtype=np.int8))
    if not blocks:
        raise RuntimeError("No positions generated")
    return scipy.sparse.vstack(blocks), np.concatenate(label_blocks), stats


def main():
    parser = argparse.ArgumentParser(
        description="Material-balanced neighbourhood dataset from novel-search "
                    "adversarial positions")
    parser.add_argument('--fens-root', type=str, default=DEFAULT_FENS_ROOT)
    parser.add_argument('--engine', type=str, default="Winter")
    parser.add_argument('--method', type=str, default="novel_mutate")
    parser.add_argument('--name', type=str, default="desk_vNovelBal")
    parser.add_argument('--out-dir', type=str, default="./../datasets_aux/")
    parser.add_argument('--tablebase', type=str, default="../../../Chess/TB_Merged")
    parser.add_argument('--target-positions', type=int, default=10_000_000,
                        help="Stop once this many positions are stored")
    parser.add_argument('--perturb-tries', type=int, default=20,
                        help="Attempts to find a valid random perturbation")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--limit', type=int, default=None,
                        help="Stop scanning after this many csv rows (smoke testing)")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    csv_files = discover_csv_files(args.fens_root, engine_glob=f"{args.engine}*",
                                   method_suffix=f"_{args.method}")
    print(f"Found {len(csv_files)} csv files")

    fens, moves, sig_codes, sig_names, rows = scan_base_positions(csv_files, limit=args.limit)
    print(f"Scanned {rows} rows: {len(fens)} unique adversarial positions")
    rank, order = balanced_order(sig_codes, sig_names, rng)
    del sig_codes

    config.tablebase = chess.syzygy.open_tablebase(args.tablebase)
    features, labels, stats = generate(fens, moves, rank, order, args.target_positions,
                                       rng, args.perturb_tries)
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
