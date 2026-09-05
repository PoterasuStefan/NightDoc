import logging
import sys

def setup_logger(name="respisense"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(module)s:%(funcName)s | %(message)s', 
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger

log = setup_logger()
