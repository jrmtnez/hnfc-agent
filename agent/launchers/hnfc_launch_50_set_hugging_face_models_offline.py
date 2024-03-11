import logging

from agent.classifiers.utils.models_cache_mgmt import refresh_cache
from agent.data.entities.config import ROOT_LOGGER_ID

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(ROOT_LOGGER_ID)
logger.setLevel(logging.INFO)


if __name__ == "__main__":
    refresh_cache()