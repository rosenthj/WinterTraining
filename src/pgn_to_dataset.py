import argparse
import chess

from data import gen_dataset_helper
import config
import count


def main():
    # Command line argument parsing
    parser = argparse.ArgumentParser(description="Script to transform pgns into datasets used for training")
    parser.add_argument('--name', type=str, default="merged")
    args = parser.parse_args()

    config.tablebase = chess.syzygy.open_tablebase("../../../Chess/TB_Merged")
    gen_dataset_helper(args.name, save=True)
    config.tablebase.close()
    print(f"TB Queries: {count.total_tb_queries}")
    print(f"Results changed by TB Query: {count.tb_changed}")

    return 0


if __name__ == '__main__':
    main()
