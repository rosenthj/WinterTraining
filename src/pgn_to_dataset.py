import argparse

from data import gen_dataset_helper


def main():
    # Command line argument parsing
    parser = argparse.ArgumentParser(description="Script to transform pgns into datasets used for training")
    parser.add_argument('--name', type=str, default="merged")
    args = parser.parse_args()

    gen_dataset_helper(args.name, save=True)

    return 0


if __name__ == '__main__':
    main()
