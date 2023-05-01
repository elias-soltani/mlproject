"""
Training different models and find the best model
"""

import os
import sys
from dataclasses import dataclass

import numpy as np

from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging

from src.utils import save_obj, evaluate_model


@dataclass
class ModelTrainerConfig:
    """ config for model"""
    trainer_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    """ model """
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (train_array[:,:-1], train_array[:,-1], test_array[:,:-1], test_array[:,-1])

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()
            }

            params = {
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                },
                "Random Forest": {'n_estimators':[8,16,32,64,128,256]},
                "Gradient Boosting": {
                    'n_estimators':[8,16,32,64,128,256],
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'subsample': [0.6,0.7,0.75,0.8,0.85,0.9],
                    },
                "Linear Regression": {},
                "K-Neighbors Regressor": {'n_neighbors':[5,7,9,11]},
                "XGBRegressor": {
                    'learning_rate':[0.1,0.01,0.05,0.001],
                    'n_estimators':[8,16,32,64,128,256],
                },
                "CatBoosting Regressor": {
                    'depth': [6,8,10],
                    'learning_rate' : [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100],
                },
                "AdaBoost Regressor": {
                    'learning_rate':[0.1,0.01,0.05,0.001],
                    'n_estimators':[8,16,32,64,128,256],}
            }

            model_report:dict = evaluate_model(X_train=X_train, y_train=y_train, X_test=X_test,
                                                y_test=y_test, models=models, params=params)
            # get the best model
            best_model_name = ""
            max_score = -np.inf

            for model_name, score in model_report.items():
                if score > max_score:
                    max_score = score
                    best_model_name = model_name

            best_model = models[best_model_name]
            if max_score <0.6:
                logging.info("Model score is less than 0.6")
                raise CustomException("Model score is -inf", sys)

            logging.info("Best model found on both training and testing dataset")

            save_obj(self.model_trainer_config.trainer_model_file_path, best_model)
            prediction = best_model.predict(X_test)
            r2_square = r2_score(y_test, prediction)
            return r2_square

        except Exception as e:
            loggin.info(e)
            raise CustomException(e, sys) from e