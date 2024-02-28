from chess_utils import get_startpos_eval, get_pos_eval  # , get_pos_semi_eval
import argparse

from data import get_max_quiet_entropy_pgn
from model import NetRelH
import os
import torch
import torch.nn as nn


def main():
    # Command line argument parsing
    parser = argparse.ArgumentParser(description="Training Script for Teacher Models")
    parser.add_argument('--pgn', type=str, default=None)
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--count', type=int, default=None)
    args = parser.parse_args()

    model = NetRelH(d=8, fd=64, num_inputs=768, activation=nn.Hardtanh(min_val=0, max_val=8))
    print("Initialized model")
    model.load_state_dict(torch.load(os.path.join("../models/", args.load), map_location=torch.device('cpu')))
    print("Loaded model")
    # entropy, fen = get_max_quiet_entropy_pgn(model, os.path.join("../pgns/", f"{args.pgn}.pgn"))
    get_max_quiet_entropy_pgn(model, os.path.join("../pgns/", f"{args.pgn}.pgn"))
    print("Returned")
    # print(fen)
    # print(entropy)


if __name__ == '__main__':
    main()
