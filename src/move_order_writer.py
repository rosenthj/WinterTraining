import argparse
from collections import OrderedDict


def load_options(option_file):
    """Return ordered dict of untrained options"""
    options = OrderedDict()
    for line in option_file.readlines():
        kv = line.split(", ")
        options[kv[0]] = int(kv[2])
    return options


def update_values(options, train_file):
    """Update option values with values returned from training"""
    for line in train_file.readlines():
        kv = line.split(", ")
        options[kv[0]] = int(kv[1].strip('\n'))
    return options


def gen_cpp(option_dict):
    """Generate C++ code to drag and drop into engine"""
    features = [
        ("Hash move", 1),
        ("Killer move", 2),
        ("Counter move", 1),
        ("Moving and target piece type", 36),
        ("Move type", 9),
        ("Move source", 10),
        ("Move destination", 10),
        ("Knight move source", 10),
        ("Knight move destination", 10),
        ("Move captures last moved piece", 1),
        ("Move SEE", 2),
        ("Move gives check", 2),
        ("Move destination is taboo", 1),
        ("Changes between non-/forcing states", 4),
        ("Rank of pawn destination", 6),
        ("Rank of passed pawn destination", 6),
        ("Pawn move attacks piece", 1),
        ("Piece is under attack", 2),
        ("History heuristic", 1),
        ("Countermove History", 2)
    ]
    f_count = 0
    f_id = 0
    print("constexpr std::array<int, 117> search_params = {")
    for i, v in enumerate(option_dict.values()):
        print(f"{4*' '}{16000 * v},{(7-len(str(v)))*' '}//{features[f_id][0]}")
        f_count += 1
        if i == 116:
            print("};")
            print("")
            print("constexpr std::array<int, 117> search_params_in_check = {")
            f_id = 0
            f_count = 0
        if features[f_id][1] <= f_count:
            f_id += 1
            f_count = 0
    print("};")


def main():
    # Command line argument parsing
    parser = argparse.ArgumentParser(description="Generate C++ code from input files")
    parser.add_argument('--all-options', type=str, default=None,
                        help="File containing SPSA inputs or uci options")
    parser.add_argument('--trained-options', type=str, default=None,
                        help="File containing SPSA trained params")
    args = parser.parse_args()
    option_filename = args.all_options
    with open(option_filename) as option_file:
        options = load_options(option_file)

    trained_options = args.trained_options
    with open(trained_options) as train_file:
        update_values(options, train_file)

    gen_cpp(options)
    return 0


if __name__ == '__main__':
    main()
