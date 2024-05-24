import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer  #the ColumnTransformer is used for creating pipeline. for onehot encoder, or any other pipleine
from sklearn.impute import SimpleImputer  #for missing values
from sklearn.pipeline import Pipeline   #to implemet the pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler  # very important because OneHotEncoder is used to handle categorical variables by converting them into a binary format, while StandardScaler is used to normalize numerical
                                                                 #features to have a mean of 0 and a standard deviation of 1. Together, they prepare the data in a format that is suitable for many machine learning algorithms.



from src.exception import CustomException  #need for handling exception
from src.logger import logging  #to log activities 
import os

from src.utils import save_object  #good for saving the pickle

#STEP 2

@dataclass  #The @dataclass decorator is required to reduce boilerplate code, improve readability, and maintainability when
# creating classes intended to store data. It automatically generates common special methods, making your code cleaner and less error-prone
class DataTransformationConfig: #We do this to hve any part that requires any input for data transformation
    preprocessor_obj_file_path=os.path.join('artifacts',"proprocessor.pkl")  # for saving any data. os.path.join is a function that helps build file paths that work correctly on any operating 
    # The code is setting up a variable called preprocessor_obj_file_path to hold the full path to a file named "preprocessor.pkl", which will be located in a folder named "artifacts".


#STEP 3

class DataTransformation:  
    def __init__(self):
        '''
        This function is responsible for data transformation
        '''
        self.data_transformation_config=DataTransformationConfig()  #first step in transformation

    def get_data_transformer_object(self):   #to create all pickle file for converting categorical features into numerical
        '''
        This function si responsible for data trnasformation
        
        '''
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]
            
            # to handle missing values, we go down to num and cat values, also doing the standard scaling. all these are hapeeniing under training data set,, and the transform is text data set
#STEP 4
            num_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")), #imputer will handle the ,missing value. we use the median as our strategy. that means we will replace all the missing values with our median
                ("scaler",StandardScaler())  #performing standard scaler.

                ]
            )
#STEP 5

            cat_pipeline=Pipeline(

                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")), #for handling missing values. the strategy we will use is most frequent, that means we will replace all our missing values with help of mode
                ("one_hot_encoder",OneHotEncoder()),           #2nd step
                ("scaler",StandardScaler(with_mean=False)) #3rd step
                ]

            )

#STEP 6

                 # write your logging info
            logging.info(f"Categorical columns: {categorical_columns}, categorical columns completed!")
            logging.info(f"Numerical columns: {numerical_columns}, categorical columns completed!")
            

               #TO COMBINE BOTH THE NUM FETAURES AND CATEGORICAL FEATURES, WE WILL USE columtransformer
#STEP 7

            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_columns), #you do it by pipeline name,what pipeline it is and our coulmns
                ("cat_pipelines",cat_pipeline,categorical_columns) #you do it by pipeline name,what pipeline it is and our coulmns

                ]


            )
#STEP 8

            return preprocessor

#STEP 9---- raise you exception 
        
        except Exception as e:
            raise CustomException(e,sys)
        
        
        
#STEP 10 ----START THE DATA TRANSFORMATION TECHNIQUE
      #1nitiate data transformation inside a function
    def initiate_data_transformation(self,train_path,test_path): # you add your self, train path and testt path. you get them from the data

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed") # to log all the details

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object() #to read all preprocessor object. this is the object created on top, too get it here.

            target_column_name="math_score"
            numerical_columns = ["writing_score", "reading_score"]

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)  #to drop columns that are not our target coulmn 
            target_feature_train_df=train_df[target_column_name]  

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1) #to drop columns that are not our target coulmn 
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)
            
        
        
# convert it into numpy.

            train_arr = np.c_[ #np.c_means numpy concentetnation
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.") #logging your information

# convert and save your file as picle file.
            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)
        
#after this you go over to utils. you can find it in src