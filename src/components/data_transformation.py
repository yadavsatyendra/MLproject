import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path:str = os.path.join("artifact","preprocessor.pkl")
    

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    def get_data_transformer_object(self):
        try:
            numerical_column=['writing_score','reading_score']
            categorical_column=['gender',"race_ethnicity","parental_level_of_education","lunch",
                                "test_preparation_course",]
            
            num_pipline= Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )
            cat_pipline=Pipeline(   
                steps=[
                    ("imputer",SimpleImputer(strategy='most_frequent')),
                    ("one_hot_encoder",OneHotEncoder(handle_unknown='ignore', sparse_output=True)),
                    

                ]
            )

            logging.info("Numerical columns standard scaling completed")
            logging.info("categorical column encoding completed")

            preprocessor=ColumnTransformer(
                [
                    ("num_pipline",num_pipline,numerical_column),
                    ("cat_pipline",cat_pipline,categorical_column)
                ]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)


    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info("read train and test data completed")
            logging.info("obtaining preprocessing object")
            preprocessing_obj=self.get_data_transformer_object()
            target_column_name="math_score"
            numerical_column=["writing_score","reading_score"]
            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]


            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]
            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            

                   # Combine the transformed data with target
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            logging.info(f"saved preprocessing object .")

            # Save the preprocessor object
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            logging.error("Error in data transformation process", exc_info=True)
            raise CustomException(e, sys)


    import os
import pickle

def save_object(file_path, obj):
    try:
        # Ensure the directory exists
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)  # Create the directory if it doesn't exist

        with open(file_path, 'wb') as file:
            pickle.dump(obj, file)
    except Exception as e:
        raise CustomException(e, sys)

    


if __name__== "__main__":
   obj=DataTransformationConfig()
   train_data,test_data= obj.initiate_data_ingestion()

   data_transformation = DataTransformation()
   data_transformation.initiate_data_transformation(train_data,test_data)

                       
