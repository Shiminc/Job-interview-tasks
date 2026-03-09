import pandas as pd
import altair as alt
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.graphics.gofplots import qqplot
from sklearn.model_selection import train_test_split
PATH = '../data/final_data.pkl'

def run_ols(X_train, X_test, y_train, y_test):
    # run ols to see how significant each variable is as scikitlearn one won't show this kind of results.
    model = sm.OLS(y_train, X_train)
    result = model.fit()
    print(result.summary())
    residuals = result.resid

    
    residuals.hist()
    plt.show()

    qqplot(residuals, line='s').show()

    return model

def preprocessing(data):

    X = pd.get_dummies(data[['mean_step_count_pre_op','mean_days_pre_op','mean_days_post_op','promis10_pre_op_mean_score', 'koosjr_pre_op_mean_score','journey_order','sex','hospital','clinician_id']],
                            columns = ['journey_order','sex','hospital','clinician_id'],
                            drop_first = True,
                            dtype = int)

    y = data[['mean_step_count_post_op']]
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test 

def main():
    # set_up_altair()
    data = pd.read_pickle(PATH)
   
    X_train, X_test, y_train, y_test = preprocessing(data)

    print('')


    run_ols(X_train, X_test, y_train, y_test)

    print('finish')

main()