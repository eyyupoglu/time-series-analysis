import pandas as pd
import requests
import matplotlib.pyplot as plt
import numpy as np

class Local_Linear_Trend():
    def __init__(self, lambda_ = 0.8, freq = '6h'):
        self.lambda_ = lambda_
        self.df_train, self.df_test = self._setTheData(freq)
        self.df_train.index = self.df_train['t']
        self.y = self.df_train['Te'].values

    def train_local_linear_trend(self):
        self.y_pred_arr, self.theta_all, self.epsilon = self._get_the_history(self.y, 5)
        self.sigma2 = _get_sigma_sqr(self.theta_all.shape[1], self.epsilon, self.lambda_)
        self._visualize_one_step_predictions()
        self._visualize_one_step_errors()

    def _get_the_history(self, y, init):
        # initialize
        F, X, epsilon, h, theta_hist, \
        theta_hat, y_pred_hist = self._get_initial_values(init, y)
        y_pred_hist[:init+1] = np.nan
        # epsilon[:init+1] = y_pred_hist[:init+1] - y[:init+1]

        # loop over observations
        for i in range(init + 1, len(y), 1):
            theta_hat, F, h = self._update(F, h, y[i], i)
            theta_hist[i, :] = theta_hat
            y_pred_hist[i] = self._predict(theta_hat, 1)
            epsilon[i] = y_pred_hist[i] - y[i]
        self.F = F
        return y_pred_hist, theta_hist, epsilon

    def predict_test(self, l):
        var_hat = self.sigma2 * (1 + f(l).T @ np.linalg.inv(self.F) @ f(l))
        mean = self._predict(self.theta_all[-1], l)
        return var_hat, mean

    def _predict(self, theta_hat, l):
        return (f(l).T @ theta_hat)

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
        y_pred = np.zeros(len(y))
        epsilon = np.zeros(len(y))
        for i in range(init):
            theta_all[i, :] = np.nan
            y_pred[i] = np.nan
            epsilon[i] = self._predict(theta_hat, 1) - self.y[i]
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

    def _update(self, F, h, yN, j):
        L = np.array([[1, 0],
                      [1, 1]])
        F = F + (self.lambda_ ** j) * f(-(j - 1)) @ f(-(j - 1)).T
        h = (self.lambda_ ** j) * np.linalg.inv(L) @ h + f(0) * yN
        theta_hat = np.linalg.inv(F) @ h
        return np.squeeze(theta_hat), F, h

    @staticmethod
    def get_estimates_LM(X, y, corr):
        return np.linalg.inv(X.T @ np.linalg.inv(corr) @ X) @ X.T @ np.linalg.inv(corr) @ y

f = lambda j: np.expand_dims(np.array([1, j]), axis=1)

def _get_sigma_sqr(p, epsilon, lambda_):
    N = len(epsilon)

    epsilon = np.expand_dims(epsilon,axis=1)
    lambda_mat = np.diag(np.ones(N))
    for i in range(N):
        lambda_mat[i,i] = lambda_**(N-i+1)
    result = epsilon.T @ lambda_mat @ epsilon / (N - p)
    return result[0][0]


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


def relaxation(X, y, n=15, cov_mat=None):
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
        return relaxation(X, y, n=n - 1, cov_mat=cov_mat)

def simulate(var, mean, N):
    n = np.random.normal(mean, var, size=N)
    plt.hist(n, alpha=0.5, bins=20, normed=True);
    plt.show();


if __name__ == '__main__':
    lltm = Local_Linear_Trend(lambda_=0.8, freq='6h')
    lltm.train_local_linear_trend()
    var, mean = lltm.predict_test(1)
    var2, mean2 = lltm.predict_test(2)
    var3, mean3 = lltm.predict_test(3)

    simulate(var, mean, 1000)
    X = _get_X_until(len(lltm.y)-1)
    cov_mat = relaxation(X, lltm.y, n=5)

    theta_hat = lltm.get_estimates_LM(X, lltm.y, cov_mat)
    predictions = np.sum(X * theta_hat.T, axis=1)

    plt.scatter(lltm.df_train['t'], lltm.y, marker='o', color='b')
    plt.scatter(lltm.df_train['t'], predictions, marker='x', color='r')
    plt.show()


    print('NP')