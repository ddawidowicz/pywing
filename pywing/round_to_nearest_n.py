def round_to_nearest_n(x, base=10):
    """
    This is a helper function that will round a value (x) to the nearest number
    specified by base. For example, round_to_nearest_n(36, base=10) -> 40
    whereas round_to_nearest_n(36, base=5) -> 35.
    ARGS:
        x       = The number to round
        base    = What number to round to, e.g. round to nearest 10
    RETURN:
        The rounded number
    """
    return int(base * round(float(x)/base))
