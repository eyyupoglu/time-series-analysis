from question_1 import *
from question_2_1 import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_estimates_LM(X, y, corr):
    return np.linalg.inv(X.T @ np.linalg.inv(corr) @ X) @ X.T @ np.linalg.inv(corr) @ y


def autocorr(x, t=1):
    return np.corrcoef(np.array([x[:-t], x[t:]]))


def cov(a, b):
    if len(a) != len(b):
        return
    a_mean = np.mean(a)
    b_mean = np.mean(b)
    sum = 0
    for i in range(0, len(a)):
        sum += ((a[i] - a_mean) * (b[i] - b_mean))
    return sum / (len(a))


def autocorr(x, t=1):
    a = x[t:]
    b = x[:-t]
    var_a = np.var(a)
    var_b = np.var(b)
    corr_ = cov(a, b / np.sqrt(var_a * var_b))
    return corr_


def reconstruct_from_rho(rho):
    cov_mat = np.diag(np.ones(len(y)))
    for i in range(cov_mat.shape[0]):
        for j in range(cov_mat.shape[1]):
            cov_mat[i, j] = rho ** (np.abs(i - j))
    return cov_mat


def relaxation(X, y, n=5, cov_mat=None):
    if n == 5:
        rho = 0
        cov_mat = np.diag(np.ones(len(y)))
    theta_hat = get_estimates_LM(X, y, cov_mat)
    predictions = np.sum(X * theta_hat.T, axis=1)
    residuals = np.squeeze(y) - predictions
    rho = autocorr(residuals, t=1)
    cov_mat = reconstruct_from_rho(rho)
    if n == 0:
        return cov_mat
    else:
        print('Rho: ', rho)
        print('Cov matrix: \n', cov_mat[:4, :4])
        return relaxation(X, y, n=n - 1, cov_mat=cov_mat)

if __name__ == '__main__':
    df3, df6 = get_data()
    df = df6

    X = df[['Te', 'Isol']].values
    X = np.append(np.expand_dims(np.ones(len(X)).T, axis=1), X, axis=1)
    y = df[['Ph']].values

    corr_mat = relaxation(X, y, n=5)
    theta_hat = get_estimates_LM(X, y, corr_mat)
    gaSol = theta_hat[2]
    Htot = -theta_hat[1]
    intercept = theta_hat[0]
    Ti = intercept / Htot

    predictions = np.sum(X * theta_hat.T, axis=1)
    sigma2 = ((np.squeeze(y) - predictions).T @ np.linalg.inv(corr_mat) @ (np.squeeze(y) - predictions))/(len(y) - 3)
    # sigma2 = np.sum(residuals ** 2) / (len(y) - 3)

    #theorem 3.2
    covar_theta = sigma2 * np.linalg.inv((X.T @ np.linalg.inv(corr_mat) @ X))
    data_sim = simulate_multivrt(covar_theta, np.squeeze(theta_hat), 1000)

    Ti_data_sim = - data_sim[0, :] / data_sim[1, :]
    m_Ti, cfDown_Ti, cfUp_Ti = get_mean_confidence_interval(Ti_data_sim)

    Htot_data_sim = data_sim[1, :]
    m_Htot, cfDown_Htot, cfUp_Htot = get_mean_confidence_interval(- Htot_data_sim)

    gaSol_data_sim = data_sim[2, :]
    m_gaSol, cfUp_gaSol, cfDown_gaSol = get_mean_confidence_interval(gaSol_data_sim)

    residuals = (np.squeeze(y) - predictions).T @ np.linalg.inv(corr_mat)

    plt.scatter(np.arange(0, len(df['t'].values)/2, 0.5), residuals)
    plt.title('Residuals 3h average dataset' )
    plt.show()


######################
    import numpy as np
    import pylab
    import scipy.stats as stats

    stats.probplot(residuals, dist="norm", plot=pylab)
    pylab.show()

    print('NP')


