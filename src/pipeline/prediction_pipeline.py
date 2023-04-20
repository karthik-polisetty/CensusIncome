import sys
import os
import pandas as pd
import numpy as np
from src.utils import load_object
from src.exception import CustomException
from src.logger import logging
from src.utils import preprocess_dataset

class PredictPipeline:
    def __init__(self):

        pass
    def predict(self,features):
        try:
            model_path = os.path.join('artifacts','model.pkl')
            preprocessor_path = os.path.join('artifacts','preprocessor.pkl')
            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)

            data = preprocess_dataset(features)
            scaled_df = preprocessor.transform(data)
            pred_value = model.predict(scaled_df)
            return pred_value
        except Exception as e:
            raise CustomException(e,sys)


class CustomData:
    def __init__(self, age, workclass, final_weight, education, education_num,
                 marital_status, occupation, relationship, race, sex, capital_gain,
                 capital_loss, hours_per_week, native_country):
        self.age = age
        self.workclass = workclass
        self.final_weight = final_weight
        self.education = education
        self.education_num = education_num
        self.marital_status = marital_status
        self.occupation = occupation
        self.relationship = relationship
        self.race = race
        self.sex = sex
        self.capital_gain = capital_gain
        self.capital_loss = capital_loss
        self.hours_per_week = hours_per_week
        self.native_country = native_country
        

    def get_data_as_dataframe(self):
        try:
            custom_input_data_dict = {
                'age': [self.age],
                'workclass': [self.workclass],
                'final_weight': [self.final_weight],
                'education': [self.education],
                'education-num': [self.education_num],
                'marital_status': [self.marital_status],
                'occupation': [self.occupation],
                'relationship': [self.relationship],
                'race': [self.race],
                'sex': [self.sex],
                'capital-gain': [self.capital_gain],
                'capital-loss': [self.capital_loss],
                'hours-per-week': [self.hours_per_week],
                'native-country': [self.native_country]
                }


            return pd.DataFrame(custom_input_data_dict)
        except Exception as e:
            raise CustomException(e,sys)
