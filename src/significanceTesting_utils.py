import numpy as np
from scipy.stats import f
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def f_test_variables(model_dict):
    y_pred = model_dict.get('y_pred')
    y_test = model_dict.get('y_test')
    x_test = model_dict.get('x_test')



    residuals = y_test - y_pred

    #residual sum of squares
    rss = np.sum(residuals**2)

    
    n= len(y_test)
    p= x_test.shape[1]

    degreesOfFreedom =  n-p


    fDict = {}
    fDict['rss'] = rss
    fDict['p'] = p
    fDict['dof'] = degreesOfFreedom



    return fDict

    
def f_test(model_dict_1, model_dict_2):
    '''
    model_dict_1: null hypthosis model
    model_dict_2: test model

    source * https://sites.duke.edu/bossbackup/files/2013/02/FTestTutorial.pdf
    '''

    #get variable Dicts
    m1 = f_test_variables(model_dict_1)
    m2 = f_test_variables(model_dict_2)

    if m1['dof'] - m2['dof'] == 0:
        f_stat = m1['rss'] / m2['rss']
    else:
        f_stat = ( (m1['rss'] - m2['rss']) / (m1['dof'] - m2['dof']) ) / (m2['rss'] / m2['dof'])

    p_value = 1 - f.cdf(f_stat, m1['dof'], m2['dof'])

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
                f_test_df.at[m1, m2] = float(fDict.get('f_stat'))
                p_val_df.at[m1,m2] = float(fDict.get('p_value'))

            else:
                f_test_df.at[m1, m2] = np.nan
                p_val_df.at[m1,m2] = np.nan


    returnDict = {}
    returnDict['f_stats'] = f_test_df
    returnDict['p_values'] = p_val_df
    f_test_df = f_test_df.apply(pd.to_numeric, errors='coerce')
    p_val_df = p_val_df.apply(pd.to_numeric, errors='coerce')
    print(p_val_df.dtypes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(p_val_df, cmap="YlGnBu", annot=True,)
    plt.title("Model Significance Testing f_test P-Values Heatmap")
    plt.xticks(rotation=45)
    plt.yticks(rotation=90)
    plt.xlabel("Test Model")
    plt.ylabel("Null Model")
    plt.savefig('figs/model_significance_testing_p_vals.png', bbox_inches='tight')
    plt.show()
    return returnDict