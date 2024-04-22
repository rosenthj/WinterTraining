import os
import config
import torch


def string_to_result_class(result_str):
    if not isinstance(result_str, str):
        return result_str
    if result_str == '1-0':
        return 0
    elif result_str == '1/2-1/2':
        return 1
    elif result_str == '0-1':
        return 2
    raise ValueError(f"Unexpected value: {result_str}")


def flip_result(result):
    return 2 - result


def init_log_file(filename: str = "log.txt", mode="a", log_path="../logs/"):
    assert config.log_file is None
    if config.name is None:
        config.log_file = open(os.path.join(log_path, filename), mode)
    else:
        config.log_file = open(os.path.join(log_path, f"{config.name}_{filename}"), mode)


def close_log_file():
    if config.log_file is not None:
        config.log_file.close()
        config.log_file = None


def log(msg: str) -> None:
    print(msg)
    if config.log_file is None:
        init_log_file()
    config.log_file.write(msg + "\n")
    config.log_file.flush()


def entropy(outcome):
    assert outcome.min() >= 0 and outcome.max() <= 1.0,\
        f"Outcome is not a probability! Values are between ({outcome.min}, {outcome.max})"
    return -(outcome * outcome.log()).sum()


def load_weights(model, name, ep, device="cpu"):
    model.load_state_dict(torch.load(f"../models/{name}/{name}_ep{ep}.pt", map_location=device))
