#STEP 1 import all libraries
import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models


#SETP 2
# CREATE MODE;L TRAINIG CONFIG

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")
    
#STEP 3
#model trainier for training the model

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

#start the model training 
 #it needs training array, test array, preprocessor path
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1], #this means take out the last column and store everythiing into x_train
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            
#STEP 4
#Try out as much model as possible, for this video will be using these few basic models to train
                                         #now we woont perform hyper-parameter tunning, but assignment, try hyper parameter tunning.
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
#STEP 5:
#Try to create parameters for each of the models
            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }
#STEP 6 
#Feed the modelsand parameters into the model report, include all parameters needed.
            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models,param=params)  #the evaluate model will be a function i will create in the utils
  
  
#STEP 7          
            # To get best model score from dict
            best_model_score = max(sorted(model_report.values()))


#STEP 8 --- Remember, the KEY is the best model name
            ## To get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

#STEP 9
#To know if the model exceeds the expectation
            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

#SETP 10
#Save the model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )#object is the best model

# STEP 11
#to see the predictied output 
            predicted=best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted) # to see how much our accuracy is
            return r2_square
            


#STEP 12
#Write your exception
        except Exception as e:
            raise CustomException(e,sys)
        
#step 13------> go to dataingestion
 #import -from src.components.model_trainer import ModelTrainerConfig
# and     from src.components.model_trainer import ModelTrainer

#STEP 14-----------> INTO DATA INGESTION
# data_transformation=DataTransformation()#data transformation
#train_arr,test_arr, = data_transformation.initiate_data_transformation(train_data, train_data)

#STEP 15 --------->Call modeltrainer
#modeltrainer=ModelTrainer()
#print(modeltrainer.initate_model_trainer(train_arr,test_arr))---->to give us our R2 score
