import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

import torch
import torch.nn as nn

from utils.types import *


def mse(y: TorchTensor, y_hat: TorchTensor) -> TorchTensor:
    """Mean Squared Error"""
    return nn.MSELoss()(y, y_hat)
    
    
def rmse(y: TorchTensor, y_hat: TorchTensor) -> TorchTensor:
    """Root Mean Squared Error"""
    return torch.sqrt(mse(y, y_hat))
    
    
def mae(y: TorchTensor, y_hat: TorchTensor) -> TorchTensor:
    """Mean Absolute Error"""
    return nn.L1Loss()(y, y_hat)
    
    
def mape(y: TorchTensor, y_hat: TorchTensor) -> TorchTensor:
    """Mean Absolute Percentage Error"""
    return torch.mean((y - y_hat).abs() / (y.abs()))


def get_errors(y: TorchTensor, y_hat: TorchTensor) -> Dict[str, float]:
    return {"mse": float(mse(y, y_hat)),
            "mae": float(mae(y, y_hat)),
            "rmse": float(rmse(y, y_hat)),
            "mape": float(mape(y, y_hat))}


def stack_errors(outputs: List[Dict[str, float]]) -> Dict[str, float]:
    n = len(outputs)
    return {"mse":sum([x['mse'] for x in outputs])/n,
            "mae": sum([x['mae'] for x in outputs])/n,
            "rmse": sum([x['rmse'] for x in outputs])/n,
            "mape": sum([x['mape'] for x in outputs])/n}
    

def DM(p_real, p_pred_1, p_pred_2, norm=2, version='univariate'):
    """
    Diebold-Mariano (DM) test.
    source: https://github.com/jeslago/epftoolbox/blob/master/epftoolbox/evaluation/_dm.py#L155
    """
 
    # Checking that all time series have the same shape
    if p_real.shape != p_pred_1.shape or p_real.shape != p_pred_2.shape:
        raise ValueError('The three time series must have the same shape')

    # Ensuring that time series have shape (n_days, n_prices_day)
    if len(p_real.shape) == 1 or (len(p_real.shape) == 2 and p_real.shape[1] == 1):
        raise ValueError('The time series must have shape (n_days, n_prices_day')

    # Computing the errors of each forecast
    errors_pred_1 = p_real - p_pred_1
    errors_pred_2 = p_real - p_pred_2

    # Computing the test statistic
    if version == 'univariate':
        # Computing the loss differential series for the univariate test
        if norm == 1:
            d = np.abs(errors_pred_1) - np.abs(errors_pred_2)
        if norm == 2:
            d = errors_pred_1**2 - errors_pred_2**2

        # Computing the loss differential size
        N = d.shape[0]

        # Computing the test statistic
        mean_d = np.mean(d, axis=0)
        var_d = np.var(d, ddof=0, axis=0)
        DM_stat = mean_d / np.sqrt((1 / N) * var_d)

    elif version == 'multivariate':

        # Computing the loss differential series for the multivariate test
        if norm == 1:
            d = np.mean(np.abs(errors_pred_1), axis=1) - np.mean(np.abs(errors_pred_2), axis=1)
        if norm == 2:
            d = np.mean(errors_pred_1**2, axis=1) - np.mean(errors_pred_2**2, axis=1)

        # Computing the loss differential size
        N = d.size

        # Computing the test statistic
        mean_d = np.mean(d)
        var_d = np.var(d, ddof=0)
        DM_stat = mean_d / np.sqrt((1 / N) * var_d)
        
    p_value = 1 - stats.norm.cdf(DM_stat)

    return p_value


def dm_test(real_price, forecasts, norm=2, title='dmtest', path='./assets/', savefig=False):
    """
    source: https://github.com/jeslago/epftoolbox/blob/master/epftoolbox/evaluation/_dm.py#L155
    """
    # Computing the multivariate DM test for each forecast pair
    p_values = pd.DataFrame(index=forecasts.columns, columns=forecasts.columns) 

    for model1 in forecasts.columns:
        for model2 in forecasts.columns:
            # For the diagonal elemnts representing comparing the same model we directly set a 
            # p-value of 1
            if model1 == model2:
                p_values.loc[model1, model2] = 1
            else:
                p_values.loc[model1, model2] = DM(p_real=real_price.values.reshape(-1,4), 
                                                  p_pred_1=forecasts.loc[:, model1].values.reshape(-1,4), 
                                                  p_pred_2=forecasts.loc[:, model2].values.reshape(-1,4), 
                                                  norm=norm, version='multivariate')

    # Defining color map
    red = np.concatenate([np.linspace(0, 1, 50), np.linspace(1, 0.5, 50)[1:], [0]])
    green = np.concatenate([np.linspace(0.5, 1, 50), np.zeros(50)])
    blue = np.zeros(100)
    rgb_color_map = np.concatenate([red.reshape(-1, 1), green.reshape(-1, 1), 
                                    blue.reshape(-1, 1)], axis=1)
    rgb_color_map = mpl.colors.ListedColormap(rgb_color_map)

    # Generating figure
    plt.imshow(p_values.astype(float).values, cmap=rgb_color_map, vmin=0, vmax=0.1)
    plt.xticks(range(len(forecasts.columns)), forecasts.columns, rotation=90.)
    plt.yticks(range(len(forecasts.columns)), forecasts.columns)
    plt.plot(range(p_values.shape[0]), range(p_values.shape[0]), 'wx')
    plt.colorbar().set_label('p-values')
    plt.title(title)
    plt.tight_layout()

    if savefig:
        plt.savefig(path + title + '.png', dpi=300)

    plt.show()