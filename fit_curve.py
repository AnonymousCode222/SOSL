import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.font_manager
import statsmodels.api as sm
 
def set_font(font_path='./HelveticaNeue.ttc'):
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['YourFontName'] + plt.rcParams['font.serif']
    plt.rcParams["font.weight"] = "normal"
    matplotlib.font_manager.fontManager.addfont(font_path)

def filter_outliers(y, x, num, threshold):
    outliers = [i for i in range(num, len(y) - num) if np.abs(y[i] - np.mean(y[i-num:i+num])) > threshold]
    return np.delete(x, outliers), np.delete(y, outliers)

def get_tokens(target_value, x, y):
    diff = np.abs(y - target_value)
    index = np.argmin(diff)
    return x[index]


def func(l, a, b, c):
    return a * np.log(l) + b * l + c

def sample(x, y, upper_value, lower_value):
    sampled_x = [x[idx] for idx in range(len(x)) if lower_value <= y[idx] <= upper_value]
    sampled_y = [y[idx] for idx in range(len(y)) if lower_value <= y[idx] <= upper_value]
    return sampled_x, sampled_y


def smooth(x, y, window_size):
    window = np.ones(window_size) / window_size
    return x[window_size-1:], np.convolve(y, window, mode='valid')
 
def plot_heatmap(data, xticklabels, yticklabels, title, xlabel, ylabel):
    ax = sns.heatmap(data, annot=True, fmt=".2f", linewidth=.5, yticklabels=yticklabels, xticklabels=xticklabels)
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.tick_params(axis='both', labelsize=10)
    plt.title(title)
 
def plot_scatter(x, y, label, xlabel, ylabel, title):
    plt.scatter(x, y, s=3, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
 
def plot_curve(x, y, label, xlabel, ylabel, title):
    plt.plot(x, y, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
 
def print_slope_intercept(x, y):
    coeffs = np.polyfit(np.log(x), np.log(y), deg=1)
    print('k,b:', round(coeffs[0],3), round(coeffs[1],3))
 
def read_training_loss_data(batch_size_list, lr_list, model_size, path_save, filename_prefix):
    training_loss_data = []
    for bs in batch_size_list:
        training_loss_data_bs = []
        for LR in lr_list:
            csv_name = path_save + 'llama_' + model_size + filename_prefix % (LR, bs, LR)
            loss_df = pd.read_csv(csv_name + '.csv')
            y = np.array(loss_df['Value'].values)[-1]
            training_loss_data_bs.append(y)
        training_loss_data.append(training_loss_data_bs)
    return training_loss_data


def plot_loss_vs_compute_budget(sampled_points_loss, sampled_points_FLOP, sampled_points_ratio, threshold):
    plt.figure(figsize=(20,20))
    sampled_ratio = np.arange(threshold, 200, threshold)
    cmap = plt.get_cmap('plasma_r')
    norm = plt.Normalize(min(sampled_ratio), max(sampled_ratio))
    fig, ax = plt.subplots()

    for _ratio in sampled_ratio:
        sampled_x, sampled_y = sample(sampled_points_FLOP, sampled_points_loss, _ratio + 0.5, _ratio - 0.5)
        plt.scatter(sampled_x, sampled_y, s=3, color=cmap(norm(_ratio)))
    plt.xscale('log')
    plt.yscale('log')
    ax.set_xlabel('Compute Budget', fontsize=10)
    ax.set_ylabel('Training Loss', fontsize=10)
    plt.title('Training Loss vs Compute Budget')
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
    cbar.set_label('OTR')


def linear_regression_analysis(x, y):
    X = sm.add_constant(x)
    model = sm.OLS(y, X)
    results = model.fit()
    print(results.summary())
    slope = results.params[1]
    d1 = sm.stats.DescrStatsW(y)
    print('t-statistic=%0.4f, p-value=%0.4e, df=%s' % d1.ttest_mean(results.params[0]))
    print(f"Slope: {slope:.3f}")

def main():
    set_font()
    linear_regression_analysis(x, y)

if __name__ == "__main__":
    main()
