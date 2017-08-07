import cProfile
import logging
import os
from logging import config

import yaml
from line_profiler import LineProfiler


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


def basic_logging(msg, format='%(asctime)s %(name)s - %(levelname)s - %(message)s',
                  level=logging.INFO):
    logging.basicConfig(format=format, level=level)
    return logging.info(msg)


def do_cprofile(func):
    """
    https://zapier.com/engineering/profiling-python-boss/
    :param func: 
    :return: 
    """

    def profiled_func(*args, **kwargs):
        profile = cProfile.Profile()
        try:
            profile.enable()
            result = func(*args, **kwargs)
            profile.disable()
            return result
        finally:
            profile.print_stats()

    return profiled_func


def do_profile(follow=[]):
    def inner(func):
        def profiled_func(*args, **kwargs):
            profiler = LineProfiler()
            profiler.add_function(func)
            for f in follow:
                profiler.add_function(f)
            profiler.enable_by_count()
            return func(*args, **kwargs)

        return profiled_func

    return inner
