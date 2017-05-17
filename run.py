# -*- encoding: utf-8 -*-
from sys import argv
import os
from src.pipeline_crf import main

if __name__ == '__main__':
    # os.environ['PROJECT_DIR'] = os.path.dirname(os.path.abspath(__file__))
    main(argv)
