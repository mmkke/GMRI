## Libraries
# general
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Random_Model import Random_Model

# preprocessing
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

# models
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, Lasso, Ridge, lars_path

# model selection
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

# metrics
from sklearn.metrics import mean_squared_error, r2_score


###################################################################################################################
## Modules

from eda_utils import *

###################################################################################################################
## Functions
###################################################################################################################

def get_regression_stats(X, y):
    '''
    Description: 
            Prints the statmodel summary of regression statistics.

            https://www.statsmodels.org/stable/index.html

    Params:
            X (ndarray): Feature Matrix
            y (ndarray): Target Vector
    Returns: 
            None
    '''

    # run stastmodel
    X_statsmodel = sm.add_constant(X) 
    statsmodel = sm.OLS(y, X_statsmodel).fit()  

    # results
    print(statsmodel.summary())
    # save as png
    summary_text = statsmodel.summary().as_text()
    plt.figure(figsize=(6, 4))
    plt.text(0.1, 0.95, summary_text, va='top', family='monospace')
    plt.axis('off')  # Turn off axes
    plt.savefig('figs/statmodel_summary', bbox_inches='tight')
    plt.show()

    # get residuals
    residuals = statsmodel.resid

    # Plot residuals against predicted values
    plt.scatter(statsmodel.fittedvalues, residuals)
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.axhline(y=0, color='r', linestyle='-')  
    plt.savefig('figs/statsmodel_residuals', bbox_inches='tight')
    plt.show()

###################################################################################################################

def univariate_regression(X, y):
    '''
    Description:
                Loops over all columns in Feature Materix and performs a linear regression for each. Plots the 
                regression line and residual plots.

                https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
    Params:
            X (ndarray): Feature Matrix
            y (ndarray): Target Vector
    Returns: None
    '''
    reg_dict_list = []

    for col in X:

        X_simple = X[col].values.reshape(-1, 1)

        model = LinearRegression()

        X_train, X_test, y_train, y_test = train_test_split(X_simple, y, test_size=0.2, random_state=42)

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        residuals = y_test - y_pred

        score = model.score(X_test, y_test)

        # metrics
        print('*'*50)
        print(f'Independet Varaible: {col.capitalize()}')

        # get and print the mean squared error (MSE)
        mse = mean_squared_error(y_test, y_pred)
        print("Mean Squared Error:", mse)

        r2 = r2_score(y_test, y_pred)
        print("R-squared:", r2)

        # print the intercept and coef
        print("Intercept:", model.intercept_)
        print("Coefficient:", model.coef_)



        fig, ax = plt.subplots(1, 2)
        # regression line
        ax[0].vlines(X_test, ymin=y_test, ymax=y_pred, color='k', linewidth=.5)
        ax[0].scatter(X_test, y_test, c='r', s=10)
        ax[0].plot(X_test, y_pred, c='b')
        ax[0].set_ylabel('Domestic Pollock Prices')
        ax[0].set_xlabel(col)
        ax[0].set_title('Regression Line')
        # residual plots
        ax[1].scatter(X_test, residuals)
        ax[1].axhline(0, color='red', linestyle='--')  
        ax[1].set_ylabel('Residuals')
        ax[1].set_xlabel(col)
        ax[1].set_title('Residual Plot')

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        fig.suptitle(f'Simple Linear Regression: {col}')
        plt.savefig(f'figs/univariate_{col}', bbox_inches='tight')
        plt.show();

        reg_dict_list.append({
                                'Regressor': col,
                                'R-squared': r2,
                                'Mean Square Error': mse,
                                'Intercept': model.intercept_,
                                'Coefficient': model.coef_
                                })

    results_df = pd.DataFrame(reg_dict_list)

    # plot all r-squared values
    sns.barplot(data=results_df, x='Regressor', y='R-squared')
    for i, value in enumerate(results_df['R-squared'].values):
        plt.text(i, (value + 0.005), '{:.3f}'.format(value), ha='center')
    plt.title(r'Explained Variance (R^2) for Univariate Regressions')
    plt.xticks(rotation=60)
    plt.legend()
    plt.savefig('figs/univariate_r2_values', bbox_inches='tight')
    plt.show()

    return createReturnDict(y_pred, X_test, y_test)

###################################################################################################################

def multivariate_regression(X, y):
    '''
    Description:
            Performs mutivariate regression  using sklearns LinearRegression() object and prints metrics. 
            Plots a barplot of the coeficients and and residuals plot. 

            https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
    Params:
            X (ndarray): Feature Matrix
            y (ndarray): Target Vector
    Returns: None
    '''  
    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # fit model
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    residuals = y_test - y_pred

    # evaluate 
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("R-squared:", r2)
    print("Mean Squared Error:", mse)
    print("Intercept:", model.intercept_)

    # get coef dataframe
    coef_df = pd.DataFrame(model.coef_.ravel(), X.columns, columns=['Coefficients'])
    print(coef_df)

    # plot test and train score
    # get scores
    y_pred_train = model.predict(X_train)
    train_score = r2_score(y_train, y_pred_train)
    scores = [train_score, r2]
    # plot
    plt.bar(['Train', 'Test'], scores, color=['blue', 'orange'])
    # annotate
    for i, value in enumerate(scores):
        plt.text(i, (value + 0.005), '{:.3f}'.format(value), ha='center') 
    plt.ylabel('R-squared score')
    plt.title('Train and Test Scores for Multivariate Regression')
    plt.savefig('figs/multi_reg_result', bbox_inches='tight')
    plt.show()


    # plot coef magnitudes
    coef_df['Abs_Coefficients'] = coef_df['Coefficients'].abs()
    sorted_coef_df = coef_df.sort_values(by='Abs_Coefficients', ascending=False)
    sns.barplot(sorted_coef_df['Coefficients'])
    for i, value in enumerate(sorted_coef_df['Coefficients'].values):
        plt.text(i, (value + 0.005), '{:.3f}'.format(value), ha='center')    
    plt.xlabel('Features')
    plt.ylabel('Coefficient Value')
    plt.title('Coefficients of the Linear Regression Model (Sorted by Magnitude)')
    plt.xticks(rotation=60)
    plt.savefig('figs/mulit_reg_coefs', bbox_inches='tight')
    plt.show()

    # get residuals
    residuals = y_pred - y_test

    # Plot residuals against predicted values
    plt.figure(figsize=(6,6))
    plt.scatter(y_pred, residuals)
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.axhline(y=0, color='r', linestyle='-')
    plt.savefig('figs/mulit_reg_residuals', bbox_inches='tight')
    plt.show()
    
    return createReturnDict(y_pred, X_test, y_test)




def pcr_regression(X, y, cv=5):
    '''
    Description:
            Performs pcr regression  using sklearns PCA and LInear Regression objects and prints metrics. 
            Plots a barplot of the coeficients and and test/train score plot.

            https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
            https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
    Params:
            X (ndarray): Feature Matrix
            y (ndarray): Target Vector
    Returns: None
    '''

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a pipe
    pipeline = Pipeline([
        ('pca', PCA()),
        ('regression', LinearRegression())
    ])

    # PCA values for gridsearch
    param_grid = {
        'pca__n_components': np.arange(1, X.shape[1])  # Specify the range of n_components to search over
    }

    # Grid search with 5-fold cross-validation
    grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring='r2', return_train_score=True)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    # Predict on the test data using the best model
    y_pred = best_model.predict(X_test)
    residuals = y_test-y_pred

    # Get results dict
    results_df = pd.DataFrame(grid_search.cv_results_)

    # Evaluate
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("R-squared:", r2)
    print("Mean Squared Error:", mse)
    print("Intercept:", best_model.named_steps['regression'].intercept_)
    print("Best Parameters:", grid_search.best_params_)

    # explained variance
    best_pca = best_model.named_steps['pca']
    explained_variance_ratio = best_pca.explained_variance_ratio_
    explained_variance_df = pd.DataFrame({
        'n_components': np.arange(1, len(explained_variance_ratio) + 1),
        'Percentage of Explained Variance': explained_variance_ratio
    })
    print('Explained Variance Ratios')
    print(explained_variance_df)

    # Plot the explained variance ratio
    plt.figure(figsize=(10, 6))
    # plot
    plt.bar(range(1, len(explained_variance_ratio) + 1), 
            explained_variance_ratio, alpha=0.8)
    # annot
    for i, value in enumerate(explained_variance_ratio):
        plt.text(i+1, (abs(value) + 0.005), '{:.3f}'.format(value), ha='center')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Explained Variance Ratio by Principal Component')
    plt.xticks(range(1, len(explained_variance_ratio) + 1))
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.savefig('figs/PCR_explained_variance', bbox_inches='tight')
    plt.show()

    ## plot test train scores
    plt.figure(figsize=(6,6))
    # train score
    plt.plot(results_df['param_pca__n_components'], 
             results_df[['mean_test_score']], 
             label='Test Scores', c='orange')
    # train score
    plt.plot(results_df['param_pca__n_components'], 
             results_df[['mean_train_score']], 
             label='Training Scores', c='blue')
    # +/- std
    plt.fill_between(x=results_df['param_pca__n_components'].astype(float), 
                    y1=results_df['mean_test_score'] + results_df['std_test_score'], 
                    y2=results_df['mean_test_score'] - results_df['std_test_score'], 
                    color='orange', alpha=0.5)
    # max test line
    plt.axhline(y=results_df['mean_test_score'].max(), 
                color='red', linestyle='dotted',
                label='Max Test Score')
    # max test annot
    plt.text(x=9, y=results_df['mean_test_score'].max()+0.01, 
             s=rf'Max $R^2$={r2:.3f}', c='red')
    # max test marker
    plt.scatter(x= results_df['param_pca__n_components'][results_df['mean_test_score'].idxmax()], 
                y=results_df['mean_test_score'].max(),
                marker='x', color='red')
    plt.xlabel('n_components')
    plt.ylabel(r'Score ($R^2$)')
    plt.legend()
    plt.title(r'PCR: Test and Train Explained Variance Ratio $(R^2)$')
    plt.savefig('figs/PCR_text_train', bbox_inches='tight')
    plt.show();
    return createReturnDict(y_pred, X_test, y_test)

###################################################################################################################

def ridge_regression(X, y, alphas, cv=5):
    '''
    Description:
            Performs ridge regression  using sklearns Ridge Regression object, and perfmorms a gridsearch with cross 
            validation over regularization parameter and prints metrics. Plots a barplot of the coeficients and and 
            test/train score plot.

            https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
    Params:
            X (ndarray): Feature Matrix
            y (ndarray): Target Vector
            alphas (ndarray): Alpha values for gridsearch.
            cv (int): Number of folds for cross validation.

    Returns: None
    '''
    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the model
    ridge = Ridge()

    # Define the grid search with cross-validation
    grid = GridSearchCV(estimator=ridge, param_grid={'alpha': alphas}, scoring='r2', cv=cv, return_train_score=True)

    # Fit the grid search to the data
    grid.fit(X_train, y_train)

    # Get the best alpha
    best_alpha = grid.best_params_['alpha']
    print("Best alpha:", best_alpha)

    # Refit the Ridge model with the best alpha
    ridge_best = Ridge(alpha=best_alpha)
    ridge_best.fit(X_train, y_train)

    # Get results dict
    results_df = pd.DataFrame(grid.cv_results_)

    # Evaluate the model
    y_pred = ridge_best.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("Mean Squared Error with best alpha:", mse)
    print("R-squared:", r2)

    # create coef dataframe
    coef_df = pd.DataFrame(ridge_best.coef_.ravel(), X.columns, columns=['Coefficients'])
    print(coef_df)

    # sort coeficients by magnitude
    coef_df['Abs_Coefficients'] = coef_df['Coefficients'].abs()
    sorted_coef_df = coef_df.sort_values(by='Abs_Coefficients', ascending=False)

    # coef barplot
    sns.barplot(sorted_coef_df['Coefficients'])
    # annot
    for i, value in enumerate(sorted_coef_df['Coefficients'].values):
        plt.text(i, (value + 0.005), '{:.3f}'.format(value), ha='center')
    # plot attributes 
    plt.xlabel('Features')
    plt.ylabel('Coefficient Value')
    plt.title('Coefficients of the Ridge Regression Model (Sorted by Magnitude)')
    plt.xticks(rotation=60)
    plt.savefig('figs/ridge_coef', bbox_inches='tight')
    plt.show()

    ## plot train test scores
    plt.figure(figsize=(6,6))
    # test scores
    plt.plot(results_df['param_alpha'], 
             results_df[['mean_test_score']], 
             label='Test Scores', c='orange')
    # train scores
    plt.plot(results_df['param_alpha'], 
             results_df[['mean_train_score']], 
             label='Training Scores', c='blue')
    # +/- std
    plt.fill_between(x=results_df['param_alpha'].astype(float), 
                    y1=results_df['mean_test_score'] + results_df['std_test_score'], 
                    y2=results_df['mean_test_score'] - results_df['std_test_score'], 
                    color='orange', alpha=0.5)
    # max test line
    plt.axhline(y=results_df['mean_test_score'].max(), 
                color='red', linestyle='dotted',
                label='Max Test Score')
    # max test annot
    plt.text(x=5.5, y=results_df['mean_test_score'].max()+0.01, 
             s=rf'Max $R^2$={r2:.3f}', c='red')
    # max test marker
    plt.scatter(x= results_df['param_alpha'][results_df['mean_test_score'].idxmax()], 
                y=results_df['mean_test_score'].max(),
                marker='x', color='red')
    plt.xlabel('Regularization (Alpha)')
    plt.ylabel(r'Score ($R^2$)')
    plt.legend()
    plt.title(r'Ridge: Test and Train Explained Variance Ratio $(R^2)$')
    plt.savefig('figs/ridge_text_train', bbox_inches='tight')
    plt.show();

    return createReturnDict(y_pred, X_test, y_test)

###################################################################################################################

def lasso_regression(X, y, alphas, cv=5):
    '''
    Description:
            Performs ridge regression  using sklearns Ridge Regression object, and perfmorms a gridsearch with cross 
            validation over regularization parameter and prints metrics. Plots a barplot of the coeficients and and 
            test/train score plot.

            https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
    Params:
            X (ndarray): Feature Matrix
            y (ndarray): Target Vector
            alphas (ndarray): Alpha values for gridsearch.
            cv (int): Number of folds for cross validation.

    Returns: None
    '''
    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the model
    lasso = Lasso()

    # Define the grid search with cross-validation
    grid = GridSearchCV(estimator=lasso, param_grid={'alpha': alphas}, scoring='r2', cv=cv, return_train_score=True)

    # Fit the grid search to the data
    grid.fit(X_train, y_train)

    # Get the best alpha
    best_alpha = grid.best_params_['alpha']
    print("Best alpha:", best_alpha)

    # Refit the Ridge model with the best alpha
    lasso_best = Ridge(alpha=best_alpha)
    lasso_best.fit(X_train, y_train)

    # Get results dict
    results_df = pd.DataFrame(grid.cv_results_)

    # Evaluate the model
    y_pred = lasso_best.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("Mean Squared Error with best alpha:", mse)
    print("R-squared:", r2)

    # coef dataframe
    coef_df = pd.DataFrame(lasso_best.coef_.ravel(), X.columns, columns=['Coefficients'])
    print(coef_df)

    # plot coefficient magnitude
    coef_df['Abs_Coefficients'] = coef_df['Coefficients'].abs()
    sorted_coef_df = coef_df.sort_values(by='Abs_Coefficients', ascending=False)
    sns.barplot(sorted_coef_df['Coefficients'])
    # annotation
    for i, value in enumerate(sorted_coef_df['Coefficients'].values):
        plt.text(i, (value + 0.005), '{:.3f}'.format(value), ha='center')
    plt.xlabel('Features')
    plt.ylabel('Coefficient Value')
    plt.title('Coefficients of the Lasso Regression Model (Sorted by Magnitude)')
    plt.xticks(rotation=60)
    plt.savefig('figs/lasso_coef', bbox_inches='tight')
    plt.show()

    ## plot train vs test score
    plt.figure(figsize=(6,6))
    # testing score
    plt.plot(results_df['param_alpha'], 
             results_df[['mean_test_score']], 
             label='Test Scores', c='orange')
    # training score
    plt.plot(results_df['param_alpha'], 
             results_df[['mean_train_score']], 
             label='Training Scores', c='blue')
    # +/-std
    plt.fill_between(x=results_df['param_alpha'].astype(float), 
                    y1=results_df['mean_test_score'] + results_df['std_test_score'], 
                    y2=results_df['mean_test_score'] - results_df['std_test_score'], 
                    color='orange', alpha=0.5)
    # max test line
    plt.axhline(y=results_df['mean_test_score'].max(), 
                color='red', linestyle='dotted',
                label='Max Test Score')
    # max test annot
    plt.text(x=0.3, y=results_df['mean_test_score'].max()+0.01, 
             s=rf'Max $R^2$={r2:.3f}', c='red')
    # max test marker
    plt.scatter(x= results_df['param_alpha'][results_df['mean_test_score'].idxmax()], 
                y=results_df['mean_test_score'].max(),
                marker='x', color='red')
    plt.xlabel('Regularization (Alpha)')
    plt.ylabel(r'Score ($R^2$)')
    plt.legend()
    plt.title(r'LASSO: Test and Train Explained Variance Ratio $(R^2)$')
    plt.savefig('figs/lasso_text_train', bbox_inches='tight')
    plt.show();

    return createReturnDict(y_pred, X_test, y_test)

###################################################################################################################
def lasso_path(X, y):
    '''
    Description: 
            Plots the LARS Path for LASSO Regression. 

            https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.lars_path.html
    Params:
            X (ndarray): Feature Matrix
            y (ndarray): Target Vector
    Returns: None
    '''

    X_array = X.to_numpy()
    y_array = y.to_numpy().ravel()


    ## LARS PATH
    print("Computing regularization path using the LARS ...")
    _, _, coefs = lars_path(X_array, y_array, method="lasso", verbose=True)

    # make a coef dataframe with a col for each feature
    coef_df = pd.DataFrame(coefs.T, columns=X.columns.values)

    # normalize xx value
    xx_df = pd.DataFrame()
    xx_df['xx'] = np.sum(np.abs(coefs.T), axis=1)
    xx_df['xx'] /= xx_df['xx'].iloc[-1]

    # create coef_order dict
    coef_order_dict = coef_df.apply(lambda x: x[x != 0].index[0] if any(x != 0) else np.nan).to_dict()

    # create list of features sorted by coef_order_dict mappng
    coef_order_list = sorted(coef_order_dict, key=coef_order_dict.get)
    print('Coefficient Order:')
    print(coef_order_list)

    ## plot
    plt.figure(figsize=(12, 8))

    # plot path for each coef
    for col in coef_df.columns:
        plt.plot(xx_df['xx'], coef_df[col], label=col)

    # plot vertical lines at branches
    ymin, ymax = plt.ylim()
    plt.vlines(xx_df, ymin, ymax, linestyle="dashed")

    # order the legend using coef_order_dict
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    sorted_handles = [by_label[label] for label in sorted(coef_order_dict, key=coef_order_dict.get)]
    sorted_labels = sorted(coef_order_dict, key=coef_order_dict.get)
    plt.legend(sorted_handles, sorted_labels, title='Coefficients in Descending Order')

    # plot attributes
    plt.xlabel("|coef| / max|coef|")
    plt.ylabel("Coefficients")
    plt.title("LASSO Path")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('figs/LARS_path', bbox_inches='tight')
    plt.show()


def randomModel(X, y):
    '''
    model that randomly assigns y_values based on the mean and stdv of the y_values in the training data
    only use is for significance testing
    '''

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    random_model = Random_Model()


    #fit and predict with random model
    random_model.fit(y_train)
    y_pred = random_model.predict(X_test)


    return createReturnDict(y_pred, X_test, y_test)


def createReturnDict(y_pred, x_test, y_test):
    '''
    creates a return dictionary for significance testing
    '''
    returnDict = {}
    returnDict['y_pred'] = y_pred
    returnDict['x_test'] = x_test
    returnDict['y_test'] = y_test

    return returnDict


###################################################################################################################
###################################################################################################################
if __name__ == '__main__':

    pass