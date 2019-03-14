import pandas as pd
import requests
import matplotlib.pyplot as plt
import numpy as np
from question_2_1 import *
import scipy

class Local_Linear_Trend():
    def __init__(self, lambda_ = 0.8, freq = '6h'):
        self.lambda_ = lambda_
        self.df_train, self.df_test = self._setTheData(freq)
        self.df_train.index = self.df_train['t']
        self.y_train = self.df_train['Te'].values

    def train_local_linear_trend(self):
        self.y_pred_hist, self.theta_hist, self.epsilon_hist = self._train_all_datapoints(self.y_train, 5)

        # X = _get_X_until(len(self.y_pred_hist) - 1)
        # y = self.y_train
        # self.corr_mat = get_corr_relaxation(X, y, n=5)
        self.T = self._get_total_memory(self.y_train)
        # eq. 3.44 but with memory
        self.var_hat = self._get_var_hat_local_estimator(self.theta_hist.shape[1], self.epsilon_hist, self.lambda_)
        self._visualize_one_step_predictions()
        self._visualize_one_step_errors()


    # def _get_var_hat(self, T):
    #     sigma_sqr = ((np.squeeze(self.y_train) - self.y_pred_hist).T @
    #                       np.linalg.inv(self.corr_mat) @
    #                       (np.squeeze(self.y_train) - self.y_pred_hist)) / (T - 3)
    #     return sigma_sqr

    def _get_var_hat_local_estimator(self, p, epsilon, lambda_):
        self.N = len(epsilon)
        epsilon = np.expand_dims(epsilon, axis=1)
        lambda_mat = np.diag(np.ones(self.N))
        for i in range(self.N):
            lambda_mat[i, i] = lambda_ ** (self.N - i + 1)
        result = epsilon.T @ lambda_mat @ epsilon / (self.T - p)
        return result[0][0]

    def _get_total_memory(self, y):
        time = np.arange(len(y))
        time = (np.ones(len(y)) - self.lambda_ ** time) / (1 - self.lambda_)
        T = int(time[-1])
        return T+1

    def _train_all_datapoints(self, y, init):
        # initialize
        F, X, epsilon_hist, h, theta_hist, \
        theta_hat, y_pred_hist = self._get_initial_values(init, y)

        X_ = _get_X_until(init)
        y_pred_hist[:init+1] =  np.squeeze(X_ @ theta_hat)
        # epsilon_hist[:init+1] = y_pred_hist[:init+1] - y[:init+1]

        # loop over observations
        for i in range(init + 1, len(y), 1):
            theta_hat, F, h = self._update_easy(F, h, y[i], i)
            theta_hist[i, :] = theta_hat
            y_pred_hist[i] = self._predict(theta_hat, 1)
            epsilon_hist[i] = y_pred_hist[i] - y[i]
        self.F_last = F
        return y_pred_hist, theta_hist, epsilon_hist

    def predict(self, l):
        var_hat = self.var_hat * (1 + f(l).T @ np.linalg.inv(self.F_last) @ f(l))
        mean_hat = self._predict(self.theta_hist[-1], l)
        return var_hat[0][0], mean_hat[0]

    def _predict(self, theta_hat, l):
        #equation 3.90
        return (f(l).T @ theta_hat)

    def _visualize_one_step_predictions(self):
        plt.scatter(np.arange(0, len(self.df_train['t'].values)), self.y_train, marker='o', color='b')
        plt.scatter(np.arange(0, len(self.df_train['t'].values)), self.y_pred_hist, marker='x', color='r')
        plt.title('One step predictions')
        plt.legend(('Observed', 'Predicted'))
        plt.show()

    def _visualize_one_step_errors(self):
        plt.scatter(np.arange(0, len(self.df_train['t'].values)), self.epsilon_hist, marker='o', color='b')
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
        y_pred = np.zeros(len(y))
        epsilon = np.zeros(len(y))
        for i in range(init):
            theta_all[i, :] = np.nan
            epsilon[i] = self._predict(theta_hat, 1) - self.y_train[i]
        return F, X, epsilon, h, theta_all, theta_hat, y_pred

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

    def _update_easy(self, F, h, yN, j):
        L = np.array([[1, 0],
                      [1, 1]])
        #eq 3.104
        F = F + (self.lambda_ ** j) * f(-(j)) @ f(-(j)).T
        h = (self.lambda_) * np.linalg.inv(L) @ h + f(0) * yN
        #eq 3.103
        theta_hat = np.linalg.inv(F) @ h
        return np.squeeze(theta_hat), F, h



    @staticmethod
    def get_estimates_LM(X, y, corr):
        return np.linalg.inv(X.T @ np.linalg.inv(corr) @ X) @ X.T @ np.linalg.inv(corr) @ y

f = lambda j: np.expand_dims(np.array([1, j]), axis=1)




def _get_X_until(i):
    X = np.ones((i + 1, 2))
    X[:, 1] = np.arange(-i, 1, 1)
    return X

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


def reconstruct_from_rho(rho, y):
    cov_mat = np.diag(np.ones(len(y)))
    for i in range(cov_mat.shape[0]):
        for j in range(cov_mat.shape[1]):
            cov_mat[i, j] = rho ** (np.abs(i - j))
    return cov_mat


def get_corr_relaxation(X, y, n=15, cov_mat=None):
    if n == 5:
        rho = 0
        cov_mat = np.diag(np.ones(len(y)))
    theta_hat = Local_Linear_Trend.get_estimates_LM(X, y, cov_mat)
    predictions = np.sum(X * theta_hat.T, axis=1)
    residuals = np.squeeze(y) - predictions
    rho = autocorr(residuals, t=1)
    cov_mat = reconstruct_from_rho(rho, y)
    if n == 0:
        return cov_mat
    else:
        print('Rho: ', rho)
        print('Cov matrix: \n', cov_mat[:4, :4])
        return get_corr_relaxation(X, y, n=n - 1, cov_mat=cov_mat)


def simulate(var, mean, N):
    n = np.random.normal(mean, var, size=N)
    return n

def get_mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), np.std(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

if __name__ == '__main__':

    lltm = Local_Linear_Trend(lambda_=0.8, freq='6h')
    lltm.train_local_linear_trend()

    # x = np.arange(0, len(lltm.df_train['t'].values))
    X =_get_X_until(37)
    X2 = X[:, 1] * -1
    X2 = np.flip(X2, axis=0)
    plt.scatter(X2, lltm.y_train)


    plt.plot(X @ lltm.theta_hist[-1])



    var, mean_hat = lltm.predict(1)
    data_sim1 = simulate(var, mean_hat, 1000)
    m_Ti, cfDown_Ti, cfUp_Ti = get_mean_confidence_interval(data_sim1)
    plt.errorbar(X2[-1] + 1, mean_hat, abs(cfUp_Ti - cfDown_Ti))

    var2, mean_hat2 = lltm.predict(2)
    data_sim2 = simulate(var2, mean_hat2, 1000)
    m_Ti2, cfDown_Ti2, cfUp_Ti2 = get_mean_confidence_interval(data_sim2)
    plt.errorbar(X2[-1] + 2, mean_hat2, abs(cfUp_Ti2 - cfDown_Ti2))

    var3, mean_hat3 = lltm.predict(3)
    data_sim3 = simulate(var3, mean_hat3, 1000)
    m_Ti3, cfDown_Ti3, cfUp_Ti3 = get_mean_confidence_interval(data_sim3)
    plt.errorbar(X2[-1] + 3, mean_hat3, abs(cfUp_Ti3 - cfDown_Ti3))

    var4, mean_hat4 = lltm.predict(4)
    data_sim4 = simulate(var4, mean_hat4, 1000)
    m_Ti4, cfDown_Ti4, cfUp_Ti4 = get_mean_confidence_interval(data_sim4)
    plt.errorbar(X2[-1] + 4, mean_hat4, abs(cfUp_Ti4 - cfDown_Ti4))

    plt.scatter(np.array([38, 39, 40, 41]), lltm.df_test['Te'])

    plt.title('95 percent prediction intervals')
    plt.show()







    print('NP')