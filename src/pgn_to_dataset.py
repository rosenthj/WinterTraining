import argparse
import re
import zlib

import chess
import numpy as np

from data import gen_dataset_helper
import config
import count


def main():
    # Command line argument parsing
    parser = argparse.ArgumentParser(description="Script to transform pgns into datasets used for training")
    parser.add_argument('--name', type=str, default="merged",
                        help="Base name of the input PGN and generated dataset (without extension)")
    parser.add_argument('--pgn-dir', type=str, default="./../pgns/",
                        help="Directory containing the input {name}.pgn file")
    parser.add_argument('--out-dir', type=str, default="./../datasets/",
                        help="Directory to write features_{name}.npz and targets_{name}.npz")
    parser.add_argument('--tablebase', type=str, default="../../../Chess/TB_Merged",
                        help="Path to the Syzygy tablebase directory used to correct endgame results")
    parser.add_argument('--tb-relabel-prob', type=float, default=1.0,
                        help="Probability of rewriting an endgame label from the tablebase, for endgame "
                             "positions past the entry into the endgame. The entry position is always "
                             "rewritten and its result still propagates back through the game; deeper "
                             "positions keep the game's actual outcome with probability 1 - p, retaining "
                             "the practical difficulty of converting. 1.0 relabels everything (default).")
    parser.add_argument('--out-suffix', type=str, default="",
                        help="Revision suffix appended to the dataset name only, not the PGN name: "
                             "--name desk_v311 --out-suffix a reads ../pgns/desk_v311.pgn and writes "
                             "features_desk_v311a.npz. Lowercase letters only, so that the loader's "
                             "version grammar keeps treating it as a newer revision of v311.")
    parser.add_argument('--drop-abnormal', action='store_true',
                        help="Drop games whose decisive result is not explained by mate, stalemate, "
                             "insufficient material, repetition or the 50-move rule -- i.e. time "
                             "forfeits, whose recorded result is unrelated to the position and would "
                             "propagate backwards through every position sampled from the game. "
                             "Needed for desk_v303..v317; harmless but unnecessary on clean PGNs.")
    parser.add_argument('--positions-per-game', type=int, default=0,
                        help="Keep at most this many positions from each game, drawn uniformly "
                             "from the positions it yielded; 0 (default) keeps all of them. "
                             "Use 1 to build a validation set in which no two positions share "
                             "a game, and so no two share a result -- a few hundred thousand "
                             "such positions carry far more information than the same number "
                             "drawn from a handful of games.")
    parser.add_argument('--seed', type=int, default=None,
                        help="Seed the position sampling, the horizontal/vertical flips and "
                             "the tablebase relabel draws, making the dataset reproducible. "
                             "Worth setting for a validation set, which may have to be "
                             "regenerated identically later. The seed is combined with --name, "
                             "so shards of one run stay independent of each other while each "
                             "one reproduces on its own.")
    args = parser.parse_args()

    if not 0.0 <= args.tb_relabel_prob <= 1.0:
        parser.error(f"--tb-relabel-prob must be in [0, 1], got {args.tb_relabel_prob}")
    # loader._VERSION_RE is ^(\d+)([a-z]*)$: anything else stops the dataset from being
    # recognised as a revision of its base version.
    if args.out_suffix and not re.fullmatch(r'[a-z]+', args.out_suffix):
        parser.error(f"--out-suffix must be lowercase letters only, got {args.out_suffix!r}")
    if args.positions_per_game < 0:
        parser.error(f"--positions-per-game must be >= 0, got {args.positions_per_game}")
    config.tb_relabel_prob = args.tb_relabel_prob
    config.drop_abnormal = args.drop_abnormal
    config.positions_per_game = args.positions_per_game

    if args.seed is not None:
        # Every draw in the conversion path goes through numpy's global RNG (data.py is the
        # only module that samples), so one seed here fixes the whole run. Mixing --name in
        # matters because the shards of a run are converted by identical commands differing
        # only in that name: a bare seed would hand all of them the same stream, applying
        # the same flip and the same relabel draw to the nth position of every shard.
        seed = (args.seed + zlib.crc32(args.name.encode())) % 2**32
        np.random.seed(seed)
        print(f"Seeded numpy with {seed} (--seed {args.seed} combined with name {args.name!r})")

    # Relabelling everything from the tablebase is what currently masks a forfeited game's
    # inverted result in the endgame. Keeping game outcomes instead is only safe once those
    # games are gone, so warn rather than silently produce inverted endgame labels.
    if args.tb_relabel_prob < 1.0 and not args.drop_abnormal:
        print("WARNING: --tb-relabel-prob < 1 without --drop-abnormal. Endgame positions from "
              "time-forfeited games will keep their recorded result, which for those games is "
              "unrelated to the position. Add --drop-abnormal unless this PGN is known clean.")

    config.tablebase = chess.syzygy.open_tablebase(args.tablebase)
    gen_dataset_helper(args.name, save=True, pgn_dir=args.pgn_dir, out_dir=args.out_dir,
                       out_name=args.name + args.out_suffix)
    config.tablebase.close()
    print(f"TB Queries: {count.total_tb_queries}")
    print(f"Results changed by TB Query: {count.tb_changed}")
    print(f"Games skipped (regular starting position): {count.skipped_startpos}")
    if count.rejected:
        total = sum(count.rejected.values())
        print(f"Games rejected as unusable: {total} of {count.total_games}")
        for reason, n in count.rejected.most_common():
            print(f"  {reason}: {n}")
    if count.relabelled_terminal:
        print(f"Games whose result was corrected from a terminal final position: "
              f"{count.relabelled_terminal}")
    if config.drop_abnormal:
        forfeits = count.skipped_abnormal + count.repaired_abnormal
        rate = 100.0 * forfeits / max(count.total_games, 1)
        print(f"Games ending in a time forfeit: {forfeits} of {count.total_games} ({rate:.1f}%)")
        print(f"  kept, result overridden by the tablebase: {count.repaired_abnormal}")
        print(f"  dropped, no tablebase position to anchor: {count.skipped_abnormal}")

    return 0


if __name__ == '__main__':
    main()
