from functools import wraps
from time import time


def timeit(func):

    @wraps(func)
    def _time_it(*args, **kwargs):
        start = int(round(time() * 1000))
        result = func(*args, **kwargs)
        end_ = int(round(time() * 1000)) - start
        print(
            f"Total execution time for function {func.__name__}: {end_/1000 if end_ > 0 else 0} seconds"
        )
        return result

    return _time_it
