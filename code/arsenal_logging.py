import os
import logging
from logging import config

import yaml

def setup_logging(path='logging.yaml', level=logging.INFO, env_key='LOG_CFG'):
    """
    Setup logging configuration
    """
    path = path
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        with open(path, 'rt') as f:
            conf = yaml.safe_load(f.read())
        config.dictConfig(conf)
    else:
        logging.basicConfig(level=level)


def basic_logging(msg, format = '%(asctime)s %(name)s - %(levelname)s - %(message)s', level = logging.INFO):
    logging.basicConfig(format=format, level=level)
    return logging.info(msg)