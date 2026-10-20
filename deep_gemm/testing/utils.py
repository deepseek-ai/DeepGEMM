import functools
import os
import torch
from typing import Callable

def get_arch_major() -> int:
    major, minor = torch.cuda.get_device_capability()
    return major


def get_arch_minor() -> int:
    major, minor = torch.cuda.get_device_capability()
    return minor


def get_arch_pair() -> tuple:
    return torch.cuda.get_device_capability()


def is_sm89() -> bool:
    """Ada Lovelace is the first architecture with FP8 Tensor Cores; Ampere (SM80) is not."""
    return get_arch_pair() == (8, 9)


def test_filter(condition: Callable):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if condition():
                func(*args, **kwargs)
            else:
                print(f'{func.__name__}:')
                print(f' > Filtered by {condition}')
                print()
        return wrapper
    return decorator


def ignore_env(name: str, condition: Callable):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if condition():
                saved = os.environ.pop(name, None)
                func(*args, **kwargs)
                if saved is not None:
                    os.environ[name] = saved
            else:
                func(*args, **kwargs)
                
        return wrapper
    return decorator
