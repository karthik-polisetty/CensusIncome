import os
import sys
from src.logger import logging
from src.exception import CustomException
from sklearn.preprocessing import OrdinalEncoder,StandardScaler,OneHotEncoder
from category_encoders.count import CountEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from dataclasses import dataclass
from src.utils import save_object



@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path:str = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            logging.info("Getting data transformation object")
            numerical_columns = ['age', 'final_weight', 'capital-gain', 'capital-loss', 'hours-per-week']
            count_enc_columns = ['workclass','marital_status','occupation','relationship','race','sex','native-country']
            ordinal_columns = ['education']

            logging.info(f"numerical columns are {numerical_columns}")
            logging.info(f"one hot encoding columns are {count_enc_columns}")
            logging.info(f"ordinal encoding columns are {ordinal_columns}")
            
            edu_categories = ['Preschool','1st-4th','5th-6th','7th-8th','9th','10th','11th','12th',
                             'HS-grad','Some-college','Assoc-voc','Assoc-acdm','Bachelors',
                             'Masters','Prof-school','Doctorate']
            

            # createing a pipeline for different types of columns and encoding required for them
            num_pipeline = Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())
                ]
            )

            ordinal_pipeline = Pipeline(
                steps =[
                ('imputer',SimpleImputer(strategy="most_frequent")),
                ('ordinal',OrdinalEncoder(categories=[edu_categories])),
                ('scaler',StandardScaler())
                ])
            
            cat_encoding_pipeline = Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy="most_frequent")),
                ('countEncoder',CountEncoder()),
                ('scaler',StandardScaler())]
            )
            

            preprocessor = ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_columns),
                ("ordinal_pipeline",ordinal_pipeline,ordinal_columns),
                ("cat_encoding_pipeline",cat_encoding_pipeline,count_enc_columns)

                ]
            )

            logging.info('preprocessor object is prepared and returned')

            return preprocessor




        except Exception as e:
            logging.info("Error occurred during making the transformation object")
            logging.error(e)
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            # reading the train and test datasets from the paths read from return of initiate_data_ingestion function
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head  : \n{test_df.head().to_string()}')

            logging.info('Obtaining preprocessing object')

            preprocessing_obj = self.get_data_transformation_object()

            # removing the target column from the main train and test datsets
            target_column = 'class'
            drop_columns = [target_column]

            logging.info('creating train and test df X,Y to fit and tranform later')

            input_feature_train_df = train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df=train_df[target_column]

            input_feature_test_df = test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df=test_df[target_column]

            logging.info("created input and target features test and train datframes")
            

            ## Transforming using preprocessor obj
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Concatinating the preprocessed arrays and they are returned by the function and so that they can be used in other module
            train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]
            
            # saving the preprocessing object in its path, so it can be used during prediction pipeline
            save_object(
                self.data_transformation_config.preprocessor_obj_file_path,
                preprocessing_obj
            )

            logging.info('preprocessor pickle file saved')

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )


        except Exception as e:
            logging.info("Error occurred during data transformation")
            logging.error(e)
            raise CustomException(e,sys)



