import logging
import sys


_logger = logging.getLogger('null')
_logger.addHandler(logging.NullHandler())


def logger() -> logging.Logger:
    return _logger


def setup_logger(name: str, log_file: str = 'resources/app.log', level: int = logging.DEBUG):
    global _logger

    if _logger.name != 'null':
        return

    _logger = logging.getLogger(name)
    _logger.setLevel(level)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)

    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setLevel(level)

    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    _logger.addHandler(file_handler)
    _logger.addHandler(console_handler)
