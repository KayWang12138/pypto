import logging
import os
import sys
from typing import Iterable, List, TypeVar

T_iter = TypeVar("T_iter", bound=Iterable)

loggers: List[logging.Logger] = []


def get_logger(name: str, env_name: str) -> logging.Logger:
    env_log_level_name = f"{env_name}_LOG_LEVEL"
    env_log_file_name = f"{env_name}_LOG_FILE"
    try:
        env_log_level = int(os.environ.get(env_log_level_name))
        mapping = {
            0: logging.NOTSET,
            1: logging.DEBUG,
            2: logging.INFO,
            3: logging.WARNING,
            4: logging.ERROR,
            5: logging.CRITICAL,
        }
        log_level = mapping[env_log_level]
    except (KeyError, TypeError, ValueError):
        log_level = logging.ERROR
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    if env_log_level := os.environ.get(env_log_file_name):
        handler = logging.FileHandler(env_log_level)
    else:
        handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(log_level)
    formatter = logging.Formatter("%(levelname)s %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    loggers.append(logger)
    return logger


def get_progress_iter(iterable: T_iter, env_name: str) -> T_iter:
    if os.environ.get(env_name) != "1":
        return iterable
    try:
        import tqdm
        from tqdm.contrib.logging import logging_redirect_tqdm

        def iterable_wrapper():
            with logging_redirect_tqdm(loggers):
                yield from tqdm.tqdm(iterable)

        return iterable_wrapper()
    except ModuleNotFoundError:
        return iterable
