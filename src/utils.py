import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import accuracy_score,roc_auc_score
#import dill
from sklearn.model_selection import GridSearchCV


def save_object(filepath,object):
    try:
        dir_path = os.path.dirname(filepath)

        logging.info("creating directory for saving object")
        os.makedirs(dir_path,exist_ok=True)

        with open(filepath, 'wb') as f:
            pickle.dump(object,f)
        

    except Exception as e:
        logging.info("Error occurred while saving object")
        raise CustomException(e,sys)

def evaluate_model(X_train,X_test,y_train,y_test,models,params):
    try:
        logging.info("model evaluation started")
        model_report = {}
        trained_models = []
        for i in range(len(models)):
            model_name = list(models.keys())[i]
            model = list(models.values())[i]

            gs = GridSearchCV(model,params)
            gs.fit(X_train,y_train)
            
            model.set_params(**gs.best_params_)

            model.fit(X_train, y_train)

            #y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            

            test_model_scores = [accuracy_score(y_test,y_test_pred),roc_auc_score(y_test,model.predict_proba(X_test)[:,1])]

            model_report[model_name]= test_model_scores

            logging.info(f"{model_name}= {test_model_scores}")
        logging.info("model evaluation finished")
        return model_report
    except Exception as e:
        logging.error(e)
        logging.info("Error occured while evaluating models")
        raise CustomException(e,sys)
    
def load_object(filepath):
    try:
        with open(filepath,'rb') as f:
            return pickle.load(f)
    except Exception as e:
        raise CustomException(e,sys)
    

def preprocess_dataset(df):
    df = df.replace(" ?", np.nan)
    cat_columns = list(df.select_dtypes(include = 'object').columns)
    num_columns = list(df.select_dtypes(exclude = 'object').columns)
    
    def space(i):
        if type(i)==str:
            return i.strip()
        else:
            return i
        
    
    for col in cat_columns:
        df[col]=df[col].apply(space)
    
    
    df.drop('education-num',inplace=True,axis = 1)
    df.drop_duplicates(inplace = True)
    
    return df
