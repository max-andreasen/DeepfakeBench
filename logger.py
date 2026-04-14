import logging


def create_logger(log_file):
    logger = logging.getLogger("deepfakebench")
    logger.setLevel(logging.INFO)

    # file handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)

    # console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger
