import os
import config


def string_to_result_class(result_str):
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
    config.log_file = open(os.path.join(log_path, filename), mode)


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

