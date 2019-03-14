from question_3_1 import *
import numpy as np
from question_2_1 import *
import matplotlib.pyplot as plt
def get_theta(df):
    X = df[['Te', 'Isol']].values
    X = np.append(np.expand_dims(np.ones(len(X)).T, axis=1), X, axis=1)
    y = df[['Ph']].values

    theta_hat = OLS_estimate(X, y)
    return theta_hat



if __name__ == '__main__':
    lltm = Local_Linear_Trend(lambda_=0.8, freq='6h')
    lltm.train_local_linear_trend()
    Te = lltm.y_train
    for i in range(4):
        var, mean_hat = lltm.predict(i)
        Te = np.append(Te, mean_hat)

    Isol = lltm.df_test['Isol'].values
    Te = Te[-4:]
    X = np.ones((4))
    X = np.vstack((X, Te))
    X = np.vstack((X, Isol))
    X = X.T

    theta_hat = get_theta(lltm.df_train)
    predictions = np.sum(theta_hat.T * X, axis=1)





    #Taking the training part of the GLM
    X_LM = lltm.df_train[['Te', 'Isol']].values
    X_LM = np.append(np.expand_dims(np.ones(len(X_LM)).T, axis=1), X_LM, axis=1)
    y_LM = lltm.df_train[['Ph']].values

    theta_hat_LM = OLS_estimate(X_LM, y_LM)

    predictions_LM = np.sum(theta_hat_LM.T * X_LM, axis=1)
    # visualise
    plt.scatter(np.arange(0, len(lltm.df_train['t'].values)), lltm.df_train['Ph'], marker='o', color='b')
    plt.scatter(np.arange(0, len(lltm.df_train['t'].values)), predictions_LM, marker='x', color='r')

    # visualise
    plt.axvspan(38, 42, facecolor='0.2', alpha=0.5)
    plt.scatter(np.arange(38, 38 + len(lltm.df_test['t'].values)), lltm.df_test['Ph'], marker='o', color='b')
    plt.scatter(np.arange(38, 38 + len(lltm.df_test['t'].values)), predictions, marker='x', color='r')
    plt.title('Linear Regression Model with time series estimations')
    plt.ylabel('The Heating')
    plt.show()

    print('NP')