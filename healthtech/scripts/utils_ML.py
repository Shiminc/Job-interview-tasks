import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

def create_data(data):
    X = data[['mean_step_count_pre_op','mean_days_pre_op','mean_days_post_op','promis10_pre_op_mean_score', 'koosjr_pre_op_mean_score','journey_order','sex','hospital','clinician_id']]
    y = data[['mean_step_count_post_op']]
    
    preprocessor = transform_features()

    X = preprocessor.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


    return  X_train, X_test, np.ravel(y_train), np.ravel(y_test) 


def transform_features():
    # tranforming data
    # did not use operation_date yet, could be year, month
    numeric_features = ['mean_step_count_pre_op','mean_days_pre_op','mean_days_post_op','promis10_pre_op_mean_score', 'koosjr_pre_op_mean_score']
    categorical_features = ['journey_order','sex','hospital','clinician_id']

    preprocessor = ColumnTransformer(
        transformers = [
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(), categorical_features)
        ]
    )

    return preprocessor

def run_grid_search(X_train, y_train, grid_search):
    model = grid_search.fit(X_train, y_train)
    print('print grid_search results')
    print('best params')
    print(model.best_params_)
    print('best_score')
    print(model.best_score_) 
    return model.best_estimator_



def get_predicted_train_test_from_best_model(best_model,X_train, y_train, X_test):
  
    y_test_predict = best_model.fit(X_train, y_train).predict(X_test)
    y_train_predict = best_model.fit(X_train, y_train).predict(X_train)

    return y_test_predict, y_train_predict

def run_evaluation(y_train, y_test, y_train_predict,y_test_predict):
    mse_score = mean_squared_error(y_test, y_test_predict)
    mae_score = mean_absolute_error(y_test, y_test_predict)

    print('evaluation based on test data')
    print(f'mse:  {mse_score}')
    print('evaluation based on test data')
    print(f'mae:  {mae_score}')


    mse_score = mean_squared_error(y_train, y_train_predict)
    mae_score = mean_absolute_error(y_train, y_train_predict)

    print('evaluation based on train data')
    print(f'mse:  {mse_score}')
    print('evaluation based on train data')
    print(f'mae:  {mae_score}')

