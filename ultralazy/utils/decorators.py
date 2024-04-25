import warnings
from functools import wraps


def suppress_warnings(func):
    """Decorator to suppress all warnings raised within the function."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Use warnings.catch_warnings context manager to suppress warnings temporarily
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return func(*args, **kwargs)

    return wrapper
