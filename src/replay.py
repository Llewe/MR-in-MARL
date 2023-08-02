import logging

from config import config

logging.basicConfig(level=logging.getLevelName(config.LOG_LEVEL))


if __name__ == "__main__":
    logging.info("Starting MR-in-MARL")
    logging.warning("Starting MR-in-MARL")
