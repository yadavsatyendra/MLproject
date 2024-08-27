import os
import sys
from dataclasses import  dataclass

from sklearn.metrics import mean_squared_error,r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression,Ridge,Lasso 
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","modle.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("spliting training and test input data")
            x_train,y_train,x_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Linear Regression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "Random Forest Regressor": RandomForestRegressor(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "XGB Regressor": XGBRegressor(),
                "Adaboost Regressor": AdaBoostRegressor()
            }
            param_grids = {
                "Lasso": {
                'alpha': [0.01, 0.1, 1, 10, 100]
                },
                "Ridge": {
                'alpha': [0.01, 0.1, 1, 10, 100]
                     },
                "Random Forest Regressor": {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10]
                 },
                "K-Neighbors Regressor": {
                'n_neighbors': [3, 5, 7, 9],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
                    },
                "Decision Tree": {
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10]
                },
                "XGB Regressor": {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
                    },
                "Adaboost Regressor": {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.5, 1.0]
                }
                }
            model_scores, best_fitted_model = evaluate_models(
              x_train=x_train, y_train=y_train,
              x_test=x_test, y_test=y_test,
              models=models, param_grids=param_grids
        )
        
            best_model_name, best_model = best_fitted_model

            if model_scores[best_model_name] < 0.6:
              raise CustomException("No suitable model found.")
        
            logging.info(f"Best model found on both training and testing dataset")

            save_object(
              file_path=self.model_trainer_config.trained_model_file_path,
              obj=best_model
        )

        # Predict using the best fitted model
            predicted = best_model.predict(x_test)
            r2_square = r2_score(y_test, predicted)
            return (r2_square,best_model)

        except Exception as e:
          raise CustomException(e, sys)
      
            

