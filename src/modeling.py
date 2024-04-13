## Libraries
# general
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# preprocessing
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

# models
from sklearn.linear_model import LinearRegression, Lasso, Ridge, lars_path

# model selection
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

# metrics
from sklearn.metrics import mean_squared_error, r2_score


###################################################################################################################
## Modules

from eda_utils import *
from modeling_utils import *

###################################################################################################################
## Functions

###################################################################################################################
## Main
def main():

    ## Set RC params
    set_params()

    ## Get Feature Matrix and Target Vector
    X = pd.read_csv('data/feature_matrix.csv')
    y = pd.read_csv('data/target_vector.csv')

    ## Modeling

    ### Statsmodel ANOVA
    get_regression_stats(X, y)

    ### Univariate regressions
    univariate_regression(X, y)

    ### Multivairate Regression
    multivariate_regression(X, y)

    ### PCR
    pcr_regression(X, y)

    ### Ridge
    alphas = np.logspace(-4, 1.5, 1000)
    ridge_regression(X, y, alphas)

    ### Lasso
    alphas = np.logspace(-4, 0, 1000)
    lasso_regression(X, y, alphas)
    lasso_path(X, y)

###################################################################################################################
if __name__ == '__main__':

    main()