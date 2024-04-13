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
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, Lasso, Ridge, lars_path

# model selection
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

# metrics
from sklearn.metrics import mean_squared_error, r2_score

###################################################################################################################
###################################################################################################################
## Modules

from eda_utils import *
###################################################################################################################
###################################################################################################################
## Functions

def get_regression_stats(X, y):

    # run stastmodel
    X_statsmodel = sm.add_constant(X.values) 
    statsmodel = sm.OLS(y.values, X_statsmodel).fit()  

    # results
    print(statsmodel.summary())

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

def multivariate_regression(X, y):
    
    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # fit model
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    residuals = y_test - y_pred

    score = model.score(X_test, y_test)
    # evaluate 
    print("Model score:", score)
    print("Intercept:", model.intercept_)
    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error:", mse)
    r2 = r2_score(y_test, y_pred)
    print("R-squared:", r2)

    # get coef dataframe
    coef_df = pd.DataFrame(model.coef_.ravel(), X.columns, columns=['Coefficients'])
    print(coef_df)

    # plot coef magnitudes
    coef_df['Abs_Coefficients'] = coef_df['Coefficients'].abs()
    sorted_coef_df = coef_df.sort_values(by='Abs_Coefficients', ascending=False)
    sns.barplot(sorted_coef_df['Coefficients'])
    for i, value in enumerate(sorted_coef_df['Coefficients'].values):
        plt.text(i, (value + 0.005), '{:.3f}'.format(value), ha='center')    
    plt.xlabel('Features')
    plt.ylabel('Coefficient Value')
    plt.title('Coefficients of the Linear Regression Model (Sorted by Magnitude)')
    plt.savefig('mulit_reg_coefs', bbox_inches='tight')
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
    plt.savefig('mulit_reg_residuals', bbox_inches='tight')
    plt.show()




def pcr_regression(X, y):


    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a pipe
    pipeline = Pipeline([
        ('pca', PCA()),
        ('regression', LinearRegression())
    ])

    # PCA values for gridsearch
    param_grid = {
        'pca__n_components': np.arange(2, X.shape[1])  # Specify the range of n_components to search over
    }

    # Grid search with 5-fold cross-validation
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2', return_train_score=True)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    # Predict on the test data using the best model
    y_pred = best_model.predict(X_test)
    residuals = y_test-y_pred

    # Get results dict
    results_df = pd.DataFrame(grid_search.cv_results_)

    # Evaluate
    r2 = r2_score(y_test, y_pred)
    print("R-squared:", r2)
    mse = mean_squared_error(y_test, y_pred)
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
    plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, alpha=0.8)
    for i, value in enumerate(explained_variance_ratio):
        plt.text(i+1, (abs(value) + 0.005), '{:.3f}'.format(value), ha='center')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Explained Variance Ratio by Principal Component')
    plt.xticks(range(1, len(explained_variance_ratio) + 1))
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.savefig('PCR_explained_variance', bbox_inches='tight')
    plt.show()

    # plot test train scores
    plt.figure(figsize=(6,6))
    plt.plot(results_df['param_pca__n_components'], results_df[['mean_test_score']], label='Test Scores', c='orange')
    plt.plot(results_df['param_pca__n_components'], results_df[['mean_train_score']], label='Training Scores', c='blue')
    plt.fill_between(x=results_df['param_pca__n_components'].astype(float), 
                    y1=results_df['mean_test_score'] + results_df['std_test_score'], 
                    y2=results_df['mean_test_score'] - results_df['std_test_score'], 
                    color='orange', alpha=0.5)
    plt.xlabel('n_components')
    plt.ylabel(r'Score ($R^2$)')
    plt.legend()
    plt.savefig('PCR_text_train', bbox_inches='tight')
    plt.show();

###################################################################################################################

def ridge_regressiomn(X, y):

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the model
    ridge = Ridge()

    # Params for gridsearch
    alphas = np.logspace(-4, 2, 1000)

    # Define the grid search with cross-validation
    grid = GridSearchCV(estimator=ridge, param_grid={'alpha': alphas}, scoring='r2', cv=5, return_train_score=True)

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
    print("Mean Squared Error with best alpha:", mse)
    r2 = r2_score(y_test, y_pred)
    print("R-squared:", r2)

    coef_df = pd.DataFrame(ridge_best.coef_.ravel(), X.columns, columns=['Coefficients'])
    print(coef_df)

    coef_df['Abs_Coefficients'] = coef_df['Coefficients'].abs()
    sorted_coef_df = coef_df.sort_values(by='Abs_Coefficients', ascending=False)
    sns.barplot(sorted_coef_df['Coefficients'])
    for i, value in enumerate(sorted_coef_df['Coefficients'].values):
        plt.text(i, (value + 0.005), '{:.3f}'.format(value), ha='center') 
    plt.xlabel('Features')
    plt.ylabel('Coefficient Value')
    plt.title('Coefficients of the Ridge Regression Model (Sorted by Magnitude)')
    plt.savefig('ridge_coef', bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(6,6))
    plt.plot(results_df['param_alpha'], results_df[['mean_test_score']], label='Test Scores', c='orange')
    plt.plot(results_df['param_alpha'], results_df[['mean_train_score']], label='Training Scores', c='blue')
    plt.fill_between(x=results_df['param_alpha'].astype(float), 
                    y1=results_df['mean_test_score'] + results_df['std_test_score'], 
                    y2=results_df['mean_test_score'] - results_df['std_test_score'], 
                    color='orange', alpha=0.5)
    plt.xlabel('Regularization (Alpha)')
    plt.ylabel(r'Score ($R^2$)')
    plt.legend()
    plt.savefig('ridge_text_train', bbox_inches='tight')
    plt.show();

###################################################################################################################

def lasso_regression(X, y):

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the model
    lasso = Lasso()

    # Params for gridsearch
    alphas = np.logspace(-4, 0, 1000)

    # Define the grid search with cross-validation
    grid = GridSearchCV(estimator=lasso, param_grid={'alpha': alphas}, scoring='r2', cv=5, return_train_score=True)

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
    print("Mean Squared Error with best alpha:", mse)
    r2 = r2_score(y_test, y_pred)
    print("R-squared:", r2)

    # coef dataframe
    coef_df = pd.DataFrame(lasso_best.coef_.ravel(), X.columns, columns=['Coefficients'])
    print(coef_df)

    # plot coefficient magnitude
    coef_df['Abs_Coefficients'] = coef_df['Coefficients'].abs()
    sorted_coef_df = coef_df.sort_values(by='Abs_Coefficients', ascending=False)
    sorted_coef_df['Coefficients'].plot(kind='bar', legend=False)
    plt.xlabel('Features')
    plt.ylabel('Coefficient Value')
    plt.title('Coefficients of the Lasso Regression Model (Sorted by Magnitude)')
    plt.savefig('ridge_coef', bbox_inches='tight')
    plt.show()

    # plot train vs test score
    plt.figure(figsize=(6,6))
    plt.plot(results_df['param_alpha'], results_df[['mean_test_score']], label='Test Scores', c='orange')
    plt.plot(results_df['param_alpha'], results_df[['mean_train_score']], label='Training Scores', c='blue')
    plt.fill_between(x=results_df['param_alpha'].astype(float), 
                    y1=results_df['mean_test_score'] + results_df['std_test_score'], 
                    y2=results_df['mean_test_score'] - results_df['std_test_score'], 
                    color='orange', alpha=0.5)
    plt.xlabel('Regularization (Alpha)')
    plt.ylabel(r'Score ($R^2$)')
    plt.legend()
    plt.savefig('lasso_text_train', bbox_inches='tight')
    plt.show();

###################################################################################################################
def lasso_path(X, y):

    from sklearn.linear_model import lars_path

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
    plt.savefig('LARS_path', bbox_inches='tight')
    plt.show()


###################################################################################################################
###################################################################################################################
if __name__ == '__main__':

    pass