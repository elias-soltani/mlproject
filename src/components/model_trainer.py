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
                "Catboosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()
            }

            model_report:dict = evaluate_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models)

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