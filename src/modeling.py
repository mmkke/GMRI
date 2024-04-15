## Libraries
import pandas as pd
import numpy as np

###################################################################################################################
## Modules
from eda_utils import *
from modeling_utils import *
from significanceTesting_utils import significanceTester
from modeling_utils import randomModel

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

    #Dictionary with information for significance tesing
    significanceDict = {}

    ## Modeling

    ### Random Model
    model_dict = randomModel(X,y)
    significanceDict['random'] = model_dict

    ### Statsmodel Summary
    print('*'*100)
    print('*'*100)
    print('STATSMODEL SUMMARY')
    get_regression_stats(X, y)

    ### Univariate regressions
    print('*'*100)
    print('*'*100)
    print('UNIVARIATE REGRESSIONS')
    univariate_regression(X, y)

    ### Multivairate Regression
    print('*'*100)
    print('*'*100)
    print('MULTIVARIATE REGRESSIONS')
    model_dict = multivariate_regression(X, y)
    significanceDict['Multivairate Regression'] = model_dict

    ### PCR
    print('*'*100)
    print('*'*100)
    print('PRINCIPAL COMPONENT REGRESSION')
    model_dict = pcr_regression(X, y, cv=5)
    significanceDict['PCR'] = model_dict

    ### Ridge
    print('*'*100)
    print('*'*100)
    print('RIDGE REGRESSION')
    alphas = np.logspace(-4, 1.5, 1000)
    model_dict = ridge_regression(X, y, alphas, cv=5)
    significanceDict['Ridge'] = model_dict

    ### Lasso
    print('*'*100)
    print('*'*100)
    print('LASSO REGRESSION')
    alphas = np.logspace(-4, 0, 1000)
    model_dict = lasso_regression(X, y, alphas, cv=5)
    lasso_path(X, y)
    significanceDict['Lasso'] = model_dict

    
    ### Significance Testing
    print('*'*100)
    print('*'*100)
    print('SIGNIFIGANCE TESTING')
    tests = significanceTester(significanceDict)
    print(tests['p_values'])
    


###################################################################################################################
if __name__ == '__main__':

    main()