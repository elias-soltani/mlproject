"""
utility functions
"""
import os
import sys

import numpy as np
import pandas as pd

import dill

from src.exception import CustomException
from src.logger import logging


def save_obj(file_path, obj):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'wb') as g:
            dill.dump(obj, g)
    except Exception as e:
        raise CustomException(e, sys) from e