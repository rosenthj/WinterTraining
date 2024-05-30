import argparse
from collections import OrderedDict


def load_options(option_file):
    """Return ordered dict of untrained options"""
    raise NotImplementedError


def update_values(option_dict, train_file):
    """Update option values with values returned from training"""
    raise NotImplementedError


def gen_cpp(option_dict):
    """Generate C++ code to drag and drop into engine"""
    raise NotImplementedError


def main():
    # Command line argument parsing
    parser = argparse.ArgumentParser(description="Generate C++ code from input files")
    parser.add_argument('--all-options', type=str, default=None,
                        help="File containing SPSA inputs or uci options")
    parser.add_argument('--trained-options', type=str, default=None,
                        help="File containing SPSA trained params")
    args = parser.parse_args()
    return 0


if __name__ == '__main__':
    main()
