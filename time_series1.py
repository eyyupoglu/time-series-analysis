import pandas as pd
import requests
import matplotlib.pyplot as plt
import numpy as np

class Local_Linear():
    def __init__(self, lambda_ = 0.8, freq = '6h'):
        self.lambda_ = lambda_
        self.df_train, self.df_test = self._setTheData(freq)
        self.df_train.index = self.df_train['t']
        self.y = self.df_train['Te'].values

    def run_local_linear_trend(self):
        self.y_pred_arr, self.theta_all, \
        self.epsilon, self.sigma_all = self._get_the_history(self.y, 6)

        self._visualize_one_step_predictions()
        self._visualize_one_step_errors()

    def _get_the_history(self, y, init):
        # initialize
        F, X, epsilon, h, theta_all, \
        theta_hat, sigma_all, y_pred = self._get_initial_values(init, y)
        y_pred[:init+1] = np.squeeze((X @ theta_hat))[:init+1]
        epsilon[:init+1] = y_pred[:init+1] - y[:init+1]

        # loop over observations
        for i in range(init + 1, len(y), 1):
            theta_hat, F, h = self._update_parameters(F, h, y[i], i)
            theta_all[i, :] = theta_hat
            y_pred[i] = (_get_X_until(i) @ theta_hat)[-1]
            epsilon[i] = y_pred[i] - y[i]
            sigma_all[i] = _get_sigma_sqr(theta_hat.shape[0], epsilon[:i-1], i-1, self.lambda_)
        return y_pred, theta_all, epsilon, sigma_all

    def _visualize_one_step_predictions(self):
        plt.scatter(self.df_train['t'], self.y, marker='o', color='b')
        plt.scatter(self.df_train['t'], self.y_pred_arr, marker='x', color='r')
        plt.title('One step predictions')
        plt.legend(('Observed', 'Predicted'))
        plt.show()

    def _visualize_one_step_errors(self):
        plt.scatter(self.df_train['t'], self.epsilon, marker='o', color='b')
        plt.title('One step prediction errors')
        plt.show()

    def _get_initial_values(self, init, y):
        X = _get_X_until(init)
        f = lambda j: np.expand_dims(np.array([1, j]), axis=1)
        F = np.zeros((2, 2))
        h = np.zeros((2, 1))
        for j in range(init):
            F = F + (self.lambda_**j) * f(-j) @ f(-j).T
            h = h + (self.lambda_**j) * f(-j) * y[init - j]
        theta_hat = np.linalg.inv(F) @ h
        theta_all = np.zeros((len(y), 2))
        sigma_all = np.zeros((len(y)))
        y_pred = np.zeros(len(y))
        epsilon = np.zeros(len(y))
        for i in range(init):
            theta_all[i, :] = np.nan
            y_pred[i] = np.nan
        return F, X, epsilon, h, theta_all, theta_hat, sigma_all, y_pred

    def _setTheData(self, freq):
        link_3h = 'https://raw.githubusercontent.com/eyyupoglu/time-series-analysis/master/data/house_data_3h.csv'
        link_6h = 'https://raw.githubusercontent.com/eyyupoglu/time-series-analysis/master/data/house_data_6h.csv'
        try:
            if freq == '3h':
                response = requests.get(link_3h)
                filename = "house_data_3h.csv"
                train = response.text
                with open(filename, "w") as text_file:
                    text_file.write(train)
            else:
                response = requests.get(link_6h)
                train = response.text
                filename = "house_data_6h.csv"
                with open(filename, "w") as text_file:
                    text_file.write(train)
        except:
            print('Couldnt download the data')
        df = pd.read_csv('house_data_6h.csv')
        return df[:-4], df[-4:]

    def _update_parameters(self, F, h, yN, j):
        L = np.array([[1, 0],
                      [1, 1]])
        f = lambda j: np.expand_dims(np.array([1, j]), axis=1)
        F = F + (self.lambda_ ** j) * f(-(j - 1)) @ f(-(j - 1)).T
        h = (self.lambda_ ** j) * np.linalg.inv(L) @ h + f(0) * yN
        theta_hat = np.linalg.inv(F) @ h
        return np.squeeze(theta_hat), F, h

def _get_sigma_sqr(p, epsilon, N, lambda_):
    epsilon = np.expand_dims(epsilon,axis=1)
    lambda_mat = np.diag(np.ones(N))
    for i in range(N):
        lambda_mat[i,i] = lambda_**(N-i+1)
    result = epsilon.T @ lambda_mat @ epsilon / (N - p)
    return result

def _get_X_until(i):
    X = np.ones((i + 1, 2))
    X[:, 1] = np.arange(-i, 1, 1)
    return X

if __name__ == '__main__':
    lltm = Local_Linear(lambda_=0.8, freq='6h')
    lltm.run_local_linear_trend()
    print('NP')