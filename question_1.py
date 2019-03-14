import pandas as pd
import requests
import matplotlib.pyplot as plt
import numpy as np


def get_data():
    global df3
    df6 = pd.read_csv('data/house_data_6h.csv')
    df6.index = df6['t']
    df3 = pd.read_csv('data/house_data_3h.csv')
    df3.index = df3['t']
    return df3, df6


if __name__ == '__main__':
    df3, df6 = get_data()
    df = df3

    fig, axs = plt.subplots(3, 1, sharex=True)
    # Remove horizontal space between axes
    fig.subplots_adjust(hspace=0)

    axs[0].scatter(np.arange(0, len(df['t'].values)/2, 0.5), df['Te'])
    axs[0].set_title('Outdoor temperature (degC)')
    axs[0].axvspan(37.5, 42, facecolor='0.2', alpha=0.5)

    axs[1].scatter(np.arange(0, len(df['t'].values)/2, 0.5), df['Isol'])
    axs[1].set_title('The solar radiation (W)')
    axs[1].axvspan(37.5, 42, facecolor='0.2', alpha=0.5)

    axs[2].scatter(np.arange(0, len(df['t'].values)/2, 0.5), df['Ph'])
    axs[2].set_title('The heating (W)')
    axs[2].axvspan(37.5, 42, facecolor='0.2', alpha=0.5)

    plt.show()
