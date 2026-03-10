import logging
from pathlib import Path


def setup_logger(name: str = "synthetic_datagen", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    Path("logs").mkdir(exist_ok=True)
    file_handler = logging.FileHandler("logs/run.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
