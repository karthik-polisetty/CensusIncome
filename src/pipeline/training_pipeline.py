import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


if __name__ == "__main__":
    # we start the data ingestion and it returns the train and test data path that we dived and created earlier
    obj = DataIngestion()
    train_path,test_path = obj.initiate_data_ingestion()

    # Data transformation created the preprocessor file and saves it it artifacts for future use and this function returns the preprocessed train and test arrays
    data_transformation = DataTransformation()
    train_arr,test_arr,_ = data_transformation.initiate_data_transformation(train_path, test_path)
    
    # Model gets trained with hyperparameter tuning and returns the best model and the respective metrics
    model_trainer = ModelTrainer()
    model_trainer.initiate_model_training(train_arr, test_arr)