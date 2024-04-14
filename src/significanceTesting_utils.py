import numpy as np
from scipy.stats import f
import pandas as pd

def f_test_variables(model_dict):
    y_pred = model_dict.get('y_pred')
    y_test = model_dict.get('y_test')
    x_test = model_dict.get('x_test')



    residuals = y_test - y_pred

    #residual sum of squares
    rss = np.sum(residuals**2)

    #degrees of freedom residuals
    n= len(y_test)
    p= x_test.shape[1]
    degreesOfFreedom = n -p


    fDict = {}
    fDict['rss'] = rss
    fDict['p'] = p
    fDict['dof'] = degreesOfFreedom

    return fDict

    
def f_test(model_dict_1, model_dict_2):
    '''
    model_dict_1: null hypthosis model
    model_dict_2: test model
    '''

    #get variable Dicts
    m1 = f_test_variables(model_dict_1)
    m2 = f_test_variables(model_dict_2)

    f_stat = ( (m1['rss'] - m2['rss']) / (m1['p'] - m2['p']) ) / (m2['rss'] / m2['dof'])
    p_value = f.sf(f_stat, m2['p'] - m1['p'], m2['dof'])

    returnDict = {}

    returnDict['f_stat'] = f_stat
    returnDict['p_value'] = p_value
    return returnDict


def significanceTester(modelDict):
    '''
    modelList is a list of dictionaries with model name as keys, and a dict with y_pred, y_test, x_test as vals
    '''

    modelNames = list(modelDict.keys())

    f_test_df = pd.DataFrame(index=modelNames, columns=modelNames)
    p_val_df = pd.DataFrame(index=modelNames, columns=modelNames)

    print(modelNames)
    print(f_test_df)
    print(p_val_df)

    for m1 in modelNames:
        for m2 in modelNames:
            if m1 != m2:
                print(m1)
                print(m2)
                fDict = f_test(modelDict[m1],modelDict[m2])
                f_test_df.at[m1, m2] = fDict.get('f_stat')
                p_val_df.at[m1,m2] = fDict.get('p_value')

            else:
                f_test_df.at[m1, m2] = None
                p_val_df.at[m1,m2] = None


    returnDict = {}
    returnDict['f_stats'] = f_test_df
    returnDict['p_values'] = p_val_df
    return returnDict