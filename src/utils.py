"""
utility functions
"""
import os
import sys

import numpy as np
import pandas as pd

from sklearn.metrics import r2_score

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
    

def evaluate_model(X_train,y_train, X_test, y_test, models):
    """
    Evaluate models
    """
    try:
        report = {}
        for model_name, model in models.items():
            model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_model_score

        return report
        
    except Exception as e:
        logging.info(e)
        raise CustomException(e, sys) from e
    
