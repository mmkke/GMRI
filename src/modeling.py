## Libraries
import pandas as pd
import numpy as np

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
    ridge_regression(X, y, alphas, cv=3)

    ### Lasso
    alphas = np.logspace(-4, 0, 1000)
    lasso_regression(X, y, alphas, cv=3)
    lasso_path(X, y)

###################################################################################################################
if __name__ == '__main__':

    main()