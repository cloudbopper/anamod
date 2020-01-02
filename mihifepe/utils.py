"""Common utility functions"""
import logging


def get_logger(name, filename, level=logging.INFO):
    """Return logger configure to write to filename"""
    formatting = "%(asctime)s: %(levelname)s: %(name)s: %(message)s"
    logging.basicConfig(level=level, filename=filename, format=formatting)  # if not already configured
    logger = logging.getLogger(name)
    return logger
