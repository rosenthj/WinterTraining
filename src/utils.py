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
