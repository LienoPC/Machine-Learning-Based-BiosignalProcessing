import numpy as np
import pandas
from matplotlib import pyplot as plt
from scipy.stats import pearsonr


def plot_to_file(signal, filename, title=None, xlabel='Sample', ylabel='Amplitude'):
    """
    Plots a 1D signal and saves the plot as an image file

    :param signal: Iterable of numeric values representing the signal.
    :param filename: Path (including filename) where the plot image will be saved.
    :param title: Title of the plot.
    :param xlabel: Label for the x-axis.
    :param ylabel: Label for the y-axis.
    """
    plt.figure()
    plt.plot(signal)
    if title:
        plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def read_prediction(csv_path):
    """
    Read a prediction csv file with columns ["Prediction", "Image", "Timestamp"] as a pandas dataframe and returns an array for each column
    :param csv_path: file path
    :return: predictions, image, timestamps
    """
    df = pandas.read_csv(csv_path)

    predictions = np.asarray(df["Prediction"])
    images = np.asarray(df["Image"])
    timestamps = np.asarray(df["Timestamp"])

    return predictions, images, timestamps

def read_raw_gsr(csv_path):
    """

    :param csv_path: file path
    :return:
    """
    df = pandas.read_csv(csv_path)

    predictions = np.asarray(df["Prediction"])
    images = np.asarray(df["Image"])
    timestamps = np.asarray(df["Timestamp"])

    return predictions, images, timestamps


def cross_correlation(signal_x, signal_y):
    """
    Computes Pearson correlation using scipy library
    :param signal_x:
    :param signal_y:
    :return:
    """
    correlation, _ = pearsonr(signal_x, signal_y)
    return correlation