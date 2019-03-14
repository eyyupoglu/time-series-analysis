from question_1 import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats

def OLS_estimate(X, y):
  return np.linalg.inv((X.T @ X ).astype(np.int)) @ X.T @ y

def simulate_multivrt(cov, mean, N):
    n = np.random.multivariate_normal(mean, cov, N).T
    return n

def get_mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), np.std(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

if __name__ == '__main__':
    df3, df6 = get_data()
    df = df6[:-4]

    X = df[['Te', 'Isol']].values
    X = np.append(np.expand_dims(np.ones(len(X)).T, axis=1), X, axis=1)
    y = df[['Ph']].values

    theta_hat = OLS_estimate(X, y)
    gaSol = theta_hat[2]
    Htot = - theta_hat[1]
    intercept = theta_hat[0]
    Ti = intercept / Htot
    predictions = np.sum(theta_hat.T * X, axis=1)
    residuals = df['Ph'] - predictions
    sigma2 = np.sum(residuals ** 2) / (len(y) - 3)

    #theorem 3.2
    covar_theta = sigma2 * np.linalg.inv((X.T @ X))

    data_sim = simulate_multivrt(covar_theta, np.squeeze(theta_hat), 1000)

    Ti_data_sim = - data_sim[0, :] / data_sim[1, :]
    m_Ti, cfUp_Ti, cfDown_Ti = get_mean_confidence_interval(Ti_data_sim)

    Htot_data_sim = - data_sim[1, :]
    m_Htot, cfUp_Htot, cfDown_Htot = get_mean_confidence_interval(Htot_data_sim)

    gaSol_data_sim = data_sim[2, :]
    m_gaSol, cfUp_gaSol, cfDown_gaSol = get_mean_confidence_interval(gaSol_data_sim)


    # visualise
    plt.scatter(np.arange(0, len(df['t'].values)), df['Ph'], marker='o', color='b')
    plt.scatter(np.arange(0, len(df['t'].values)), predictions, marker='x', color='r')
    plt.title('Linear Regression Model')
    plt.ylabel('The Heating')
    plt.legend(('Observed', 'Predicted'))
    plt.show()

    plt.scatter(np.arange(0, len(df['t'].values)), residuals)
    plt.title('Residuals')
    plt.show()

