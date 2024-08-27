import os
import sys
from src.logger import logging

import numpy as np
import pandas as pd
import dill
from sklearn.metrics import r2_score

from src.exception import CustomException
from sklearn.model_selection import GridSearchCV

def save_object(file_path, obj):
    try:
        dir_path=os.path.dirname(file_path)
        
        os.makedirs(dir_path,exist_ok=True)
        with open (file_path,"wb")as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)


    
def evaluate_models(x_train, y_train, x_test, y_test, models, param_grids):
    model_scores = {}
    best_fitted_model = None
    
    for model_name, model in models.items():
        try:
            logging.info(f"Training model: {model_name}")
            param_grid = param_grids.get(model_name, {})
            
            if not param_grid:
                logging.info(f"No parameter grid provided for {model_name}. Using default parameters.")
            
            grid_search = GridSearchCV(model, param_grid, cv=3, scoring='r2')
            grid_search.fit(x_train, y_train)
            
            best_model = grid_search.best_estimator_
            predictions = best_model.predict(x_test)
            score = r2_score(y_test, predictions)
            
            logging.info(f"Model: {model_name}, Score: {score}")
            model_scores[model_name] = score
            
            # Store the best fitted model if it's the best so far
            if best_fitted_model is None or score > model_scores.get(best_fitted_model[0], -1):
                best_fitted_model = (model_name, best_model)
        
        except Exception as e:
            logging.warning(f"Failed to train {model_name}: {e}")
            model_scores[model_name] = None

    if not model_scores:
        logging.error("No models were evaluated successfully.")
    
    return model_scores, best_fitted_model
