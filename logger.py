import logging


def create_logger(log_file):
    logger = logging.getLogger("deepfakebench")
    # Clear handlers from prior calls. Otherwise every create_logger() call
    # under an Optuna sweep (100s of trials) keeps stacking file + stream
    # handlers on the same shared-name logger, which leaks FDs and duplicates
    # every log line N times.
    for h in list(logger.handlers):
        logger.removeHandler(h)
        h.close()
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
    logger.propagate = False
    return logger
