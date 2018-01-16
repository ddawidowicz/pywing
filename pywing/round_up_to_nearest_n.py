def round_up_to_nearest_n(x, base=10):
    """
    This is a helper function that will round a value (x) up to the nearest
    number specified by the base. For example,
            round_to_nearest_n(36, base=10) -> 40
            round_to_nearest_n(33, base=10) -> 40.
    ARGS:
        x       = The number to round
        base    = What number to round to, e.g. round to nearest 10
    RETURN:
        The rounded number
    """
    return int(base * np.ceil(x / base))
