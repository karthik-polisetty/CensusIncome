import pandas as pd
import numpy as np
import os
import sys
from src.logger import logging
from src.exception import CustomException
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from src.utils import save_object,load_object,evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path:str = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self,train_arr,test_arr):
        try:
            logging.info('Starting model training')
            logging.info('splitting the transformed arrays into dependent and independent arrays')
            X_train = train_arr[:,0:-1]
            y_train = train_arr[:,-1]

            X_test = test_arr[:,0:-1]
            y_test = test_arr[:,-1]

            models = {
                'Logistic Regression' : LogisticRegression(),
            }

            hyperparams = {
                "penalty": ["l1", "l2"],
                "C": [0.001, 0.01, 0.1, 1, 10, 100],
                "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
                "max_iter": [100, 500, 1000]}


            model_report = evaluate_model(X_train = X_train,X_test = X_test, y_train= y_train,y_test =  y_test,models = models,params = hyperparams)

            print(model_report)
            
            print('\n====================================================================================\n')
            logging.info(f'Model Report : {model_report}')

            logging.info("model report generated")

            best_model = models["Logistic Regression"]
            
            print(f'Model trained , Model Name : Logistic Regression , accuracy score : {list(model_report.values())[0][0]}, roc_auc_score: {list(model_report.values())[0][1]}')
            print('\n====================================================================================\n')
            logging.info(f'Model trained , Model Name : Logistic Regression , accuracy score : {list(model_report.values())[0][0]}, roc_auc_score: {list(model_report.values())[0][1]}')

            save_object(
                 filepath=self.model_trainer_config.trained_model_file_path,
                 object=best_model
            )




        except Exception as e:
            raise CustomException(e,sys)

