"""
utility functions
"""
import os
import sys

import numpy as np
import pandas as pd

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

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
    

def evaluate_model(X_train,y_train, X_test, y_test, models, params):
    """
    Evaluate models
    """
    try:
        report = {}
        for model_name, model in models.items():
            model_parameters = params[model_name]
            gs = GridSearchCV(model, model_parameters, cv=3)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
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
    

def load_object(file_path):
    try:
        with open(file_path, 'rb') as f:
            return dill.load(f)
    except Exception as e:
        logging.error(e)
        raise CustomException(e, sys) from e
    
