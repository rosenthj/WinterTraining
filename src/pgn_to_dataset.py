import argparse
import chess

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
    args = parser.parse_args()

    config.tablebase = chess.syzygy.open_tablebase(args.tablebase)
    gen_dataset_helper(args.name, save=True, pgn_dir=args.pgn_dir, out_dir=args.out_dir)
    config.tablebase.close()
    print(f"TB Queries: {count.total_tb_queries}")
    print(f"Results changed by TB Query: {count.tb_changed}")
    print(f"Games skipped (regular starting position): {count.skipped_startpos}")

    return 0


if __name__ == '__main__':
    main()
