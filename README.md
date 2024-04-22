# project-spring24

**Project:** GMRI  
**Team:** 
* Ned Hallahan (NedHallahan)
* Josh Nougaret (jnougaret)
* Michael Massone (mmkke) - Team Lead


## Project Info

* Stakeholder
  * [Dr. Kanae Tokunaga](https://gmri.org/our-approach/staff/kanae-tokunaga/), Senior Scientist (ktokunaga@gmri.org),
  [Gulf of Maine Research Institute (GMRI)](http://gmri.org)
* Story
  * The GMRI Coastal and Marine Economics Lab seeks to understand the mechanisms behind
  decisions and behaviors related to coastal and marine resource uses. This project focuses on the groundfish.
  New England is home to a groundfish fishery that produces a variety of whitefish,
  including species that are culturally important, such as cod and haddock. 
  We also import a large amount of whitefish from the Northeast Atlantic. 
  In other words, locally harvested whitefish directly competes with imports from the Northeast Atlantic. 
  We want to understand the market competition and relationships between New England and Northeast Atlantic 
  (e.g., Norway, Iceland, Scotland). 
* Data
  * Global: https://www.sea-ex.com/trading/market.htm
  * Portland Fish Exchange: https://www.pfex.org/price-landing-tool/
  * NOAA Landings and Foreign Trade: https://www.fisheries.noaa.gov/foss/f?p=215:2:5473541341067


___
## Goal

To determine if imports of groundfish from countries with fisheries in the Barents Sea impact the price of domestically landed Pollock in Maine. This inquiry is guided by the premise that decreasing fish stocks and recent restrictions, which decreased the catch limits for Cod and Haddock in the Barents Sea starting in 2022, have had a positive impact on the market price of previously less desirable species like Pollock for Maine fishermen. The goal of our project is to determine what role, if any, the Barents Sea fishery plays in influencing market price for Domestic Pollock in Maine. We set out to accomplish this task with the understanding that numerous trade and energy related variables are at play within the context of a complex global market.

___
## Approach

Our approach is to perform a regression analysis on Domestic Pollock prices at the Portland Fish Exchange, incorporating the prices and quantities of various imported and domestic groundfish catches as inputs to our models. We use data provided by the Portland Fish Exchange and the NOAA Fisheries Database from 2014 to 2024. 

___
## Prep Data and Environment

Run the following make commands to setup the environment and run the src/modeling_prep.py file. This file will access the data and perform additional EDA and preprocessing for regression analysis. The Feature Matrix and Target Vector will be saved as CSV files. 

```Make
make                # environment
make model_prep     # setup, fetch data, modeling prep
```


### Format Data for Modeling

Data was imported from the NOAA Fisheries Database and the Portland Fish Exchange (PFE) and combined into a single  CSV file. Although NOAA data is reported daily, PFE data is reported monthly. For this reason, NOAA data was summarized as monthly reports. After importing basic preprocessing steps as showcased in the EDA, Price values were adjusted for inflation.

Data is accessed using the get_data() and preprocess_data() functions defined in eda_util.py. Documentation for these functions is included in EDA.md.

```python
data = get_data()             
data = preprocess_data(data)
```

### Select Countries of Interest

The data obtained from NOAA encompassed all imports from the European region. We filtered this dataset to include only countries with active Barents Sea fishery quotas, focusing on Norway and Russia, which control the majority of the harvest in the area of interest, along with Iceland, which fishes in the northern Arctic waters. We then combined these imports with domestic landings recorded in Portland, Maine, obtained from the Portland Fish Exchange.


```python
data_filtered = data[data['Country'].isin(['USA', 'ICELAND', 'NORWAY', 'RUSSIAN FEDERATION'])]
```

### Create Pivot Tables

The selected data was then formatted for regression analysis. Two pivot tables were created: one for quantity and the other for value, with table rows formatted as period indexes in the MM/YYYY format. The Portland Fish Exchange data, which reported monthly totals, served as the limiting factor in terms of the time scale. The columns in these tables represent the fish species and their origin (domestic versus imported). These two tables were subsequently merged into a single dataframe for further preprocessing.

```python
### Create Pivot Tables
# drop the 'Country' column
data_without_country = data_filtered.drop('Country', axis=1)

# create a unique identifier for each fish group by its import status
data_without_country['FishGroup_ImportStatusValue'] = np.where(data_without_country['Imported'] == 'Yes',
                                                        data_without_country['FishGroup'] + "_Imported_USD",
                                                        data_without_country['FishGroup'] + "_Domestic_USD")

# create a unique identifier for each fish group by its import status
data_without_country['FishGroup_ImportStatusAmount'] = np.where(data_without_country['Imported'] == 'Yes',
                                                        data_without_country['FishGroup'] + "_Imported_Kilos",
                                                        data_without_country['FishGroup'] + "_Domestic_Kilos")

# pivot the table to have dates as rows and the unique fish group import statuses as columns, with average prices as values
df_value = data_without_country.pivot_table(index='YYYY/MM', 
                                            columns='FishGroup_ImportStatusValue', 
                                            values='AvgPrice_per_Kilo',
                                            aggfunc='mean')
# pivot the table to have dates as rows and the unique fish group import statuses as columns, with average prices as values
df_amount = data_without_country.pivot_table(index='YYYY/MM', 
                                            columns='FishGroup_ImportStatusAmount', 
                                            values='AmountSold_by_Kilo',
                                            aggfunc='sum')
```



### Check Distribution of Features

The feature columns were plotted as histograms to visually assess the Price and Volume distributions. 

<img src="figs/value_dist.png" width=800>
<img src="figs/amount_dist.png" width=800>

### Transform Volume Features

The Price columns appeared to be close to normally distributed. However, the quantity distributions appeared skewed so we implemented a log transformation to try to produce more normally distributed data. The results of that transformation can be seen below. 

<img src="figs/amount_log_dist.png" width=800>

### Set Time Frame

We selected a 10-year timeframe from 01/2014 to 01/2024. This timeframe was chosen to encompass all available data following the 2012 declaration of the Northeast Multispecies Groundfish Fishery as a fishery disaster area, which led to significant changes in Import trends. Our focus was specifically on trade and price dynamics from countries fishing in the Barents Sea within the post-2013 paradigm which was highlighted by our EDA.

```python
### Set Time Frame
start_period = pd.Period('2014-01', freq='M')
end_period = pd.Period('2024-04', freq='M')
filtered_df_range = df_combined[(df_combined.index >= start_period) & (df_combined.index <= end_period)].copy()
```

### Drop NaN Values and Impute

At this point the data included many NaN values that needed to be resolved before regression could be performed. We used the Missingno library to visually assess the ratio and distribution of NaN values within the dataset.  

<img src="figs/missingness1.png" width=800>

Some of the less relevant columns with a high number of NaN values were dropped completely. These included:

['Hake_Imported_USD', 'Hake_Imported_Kilos', 'Redfish_Imported_USD', 'Redfish_Domestic_USD', 'Redfish_Domestic_Kilos', 'Redfish_Imported_Kilos']

Since the main focus of our inquiry was on Haddock, Cod and Pollock, it seemed unnecessary to keep these features in the analysis with such a large percentage of NaN values.

<img src="figs/missingness2.png" width=800>

The Imported Pollock data also included NaN values, but these seemed more likely to have a strong bearing on our analysis. We visualized the distributions to insure a normal-like distribution, then imputed the values using the mean of the respective data. 

<img src="figs/dist_imputed_values.png" width=400><img src="" width=400>

Lastly we eliminated any remaining rows with NaN values. 

<img src="figs/missingness3.png" width=800>

The final result was a Feature Matrix with 120 rows and 13 features, with each row representing one month over the 10 year period. Price data objects represent the mean price per kg for that month, and quantity data objects are the total landing of imported fish species for that month.

### Visualize Data

A quick visualization of the value and amount over time. 

<img src="figs/data_viz_value.png" width=800>  


<img src="figs/data_viz_amount.png" width=800>

We can see an increase in the price of imported Cod and Haddock over the past two years with what appears to be the start of a downward trend in the volume of the same species. This could be an indicator that our initial assumption that changes in Barents Sea fish stocks and recently decreased quotas are impacting imports to Maine. The relation of these trends to Pollock prices is not clear from this visualization.
 

Check for effects of seasonality with boxplots.

<img src="figs/amount_boxplots.png" width=800>

<img src="figs/value_boxplots.png" width=800>

There is not clear impact of seasonality on price based on this plot. There may be some effect on volume, but the relationship is not clear. 

### Correlations and Pairplots

Checking for correlation amoung the features. 

<img src="figs/heatmap.png" width=800>

There do appear to be some minor correlations between some of the features. At this point we will not drop any features from the analysis, but we will need to look into ways of accounting for potential covariances in the regression analysis. 

<img src="figs/pairplots.png" width=800>


### Scaling

We utilized the Standard Scaler from scikit-learn to normalize and mean center features prior to analysis. 


### Export Feature Matrix and Target Vector		

The feature matrix and target vector were saved as CSVs. 


___
## Modeling

Execute the following make command to run the src/modeling.py file. This file will run several regression analyses and return the results and plots. 

```Make
make modeling
```

### Statsmodel Summary

Our first step was to generate a Statmodel Summary report for the Ordinary Least Square Regression of our Features Matrix (X) on the Target Vector (y). 

<img src="figs/statmodel_summary.png" width=800>  

\
Explained Variance Ratio (R^2):


$R^2 = 1 - \frac{{\sum_{i=1}^n (y_i - \hat{y}_i)^2}}{{\sum_{i=1}^n (y_i - \bar{y})^2}} = \frac{\text{Explained Sum of Squares}}{\text{Total Sum of Squares}} = 0.518$  

This summary demonstrates that the Ordinary Least Square regression model explains about 52% of the total variance in the domestic Pollock Price. The Adjusted R-squared value indicates that some of our variables are not contributing to the overall score, since R-squared tends to increase with more features. 

F-statistic and Prob (F-statistic):

The F-statistic compares the regression model obtained to the null hypothesis (that Domestic Pollock Prices are independent of the features) where the coefficients of all variables are zero. The probability of the F-statistic gives the probability that the null hypothesis is true, given the the F-statistic. In this case, it is quite low indicating that the regression model is likely producing a significant result and will be a useful approach to pursue. 

Coefficients:

With p-value < 0.01:
```Markdown
==========================================================================================
                             coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------------------
Cod_Imported_USD          -0.3615      0.127     -2.855      0.005      -0.612      -0.110
Hake_Domestic_USD          0.4876      0.121      4.030      0.000       0.248       0.728
Haddock_Imported_Kilos    -0.5231      0.128     -4.099      0.000      -0.776      -0.270
Hake_Domestic_Kilos        0.4379      0.142      3.091      0.003       0.157       0.719
Pollock_Domestic_Kilos    -0.8893      0.178     -5.002      0.000      -1.242      -0.537
```
Imported Cod and Haddock prices do seem to have a significant effect, as predicted, on Pollock prices. There are also several domestic factors that seem to be influencing pollock price. Not surprisingly, the amount of domestic Pollock caught has a significant negative impact on price. Based on this analysis it seems that Pollock_Domestic_Kilos has the strongest impact on our dependent variable.


The plot of the residuals below shows a distribution of residuals that appears random. Again this is a good indicator that a linear model is good fit for this dataset. 

<img src="figs/statsmodel_residuals.png" width=800>

Our next steps will be to further investigate the relationship between X and y and try to find a regression model with the best possible predictive power as measured by explained variance. To do so we will look at univariate, multivariate, and PCR regressions. Additionally, we will try LASSO and Ridge Regression model to better generalize the model by regularizing the coefficients.

### Univariate Regression

Next step is to compute the univariate regression for all features in our dataset. 

<img src="figs/univariate_r2_values.png" width=800>

We can select and look at the regression line and residual plot for all features that had a significant p-value:

<img src="figs/univariate_Cod_Imported_USD.png" width=800>
<img src="figs/univariate_Haddock_Imported_Kilos.png" width=800> 
<img src="univariate_Hake_Domestic_Kilos.png" width=800>
<img src="figs/univariate_Hake_Domestic_USD.png" width=800>
<img src="figs/univariate_Pollock_Domestic_Kilos.png" width=800>

Visually there appear to be a high degree of bias in some cases. For example, the regression for Haddock_Imported_Kilos might be better represented with a polynomial regression. Since we are less concerned with individual relationships we will move on to try to improve the multivariate regression.   


### Multivariate Regression

Multivariate Regression using scikit-learn with train test split. 

Results:

```Markdown
MULTIVARIATE REGRESSIONS
R-squared: 0.5823919767940072
Mean Squared Error: 0.9472081346425588
Intercept: [4.92655315]
```

The explained variance ratio is somewhat improved over the initial Statmodel estimate. Interestingly, this model performed better on the test data than on the training data. 

<img src="figs/multi_reg_result.png" width=800>

<img src="figs/mulit_reg_coefs.png" width=800>

<img src="figs/mulit_reg_residuals.png" width=600>


### PCR

Since we have 13 features attempting some dimensionality reduction is another possible avenue for improving our regression analysis. In this analysis we performed a Principle Component Regression, using a Gridsearch with cross-validation over the number of components.

**Results:**
```Markdown
PRINCIPAL COMPONENT REGRESSION
R-squared: 0.5221034655154911
Mean Squared Error: 1.0839530368838926
Intercept: [4.89311077]
Best Parameters: {'pca__n_components': 11}
```

<img src="figs/PCR_explained_variance.png" width=800>

Performing a Gridsearch over the value of n_components failed to produce a low dimensional embedding of the data that retains most of the variance. Based on this model, the variance if distributed across the feature space. 

**Grid Search over n_components with 5 Fold Cross Validation**  

<img src="figs/PCR_text_train.png" width=600>

The Gridsearch found the max mean cross validation score with 11 components. 


### Ridge

THe Ridge regression performs shrinkage by reducing the impact of low performing variables on the result. It does so by applying a penalty term $(\lambda)$. THe value of the penalty is determined by the alpha, over which we will perform a Gridsearch with cross validation, to find the best performing value of alpha. A value of 0 indicates no shrinkage of the coefficients. As the alpha increase so does the amount of shrinkage or regularization. 

Object Function for Ridge:

$ \text{minimize} \left( \frac{1}{2n} \sum_{i=1}^{n} (y_i - \beta_0 - \sum_{j=1}^{p} \beta_j x_{ij})^2 + \lambda \sum_{j=1}^{p} \beta_j^2 \right) $

where:

- $(n)$ is the number of samples,
- $(p)$ is the number of features,
- $(y_i)$ is the target value for the $(i)$th sample,
- $(x_{ij})$ is the value of the $(j)$th feature for the $(i)$th sample,
- $(\beta_0, \beta_1, \ldots, \beta_p)$ are the coefficients,
- $(\lambda)$ is the regularization parameter.


**Results:**
```Markdown
RIDGE REGRESSION
Best alpha: 16.357378899298137
Mean Squared Error with best alpha: 1.026846438709015
R-squared: 0.5472807974065743
```

<img src="figs/ridge_coef.png" width=800>

**Grid Search over Alpha with 5 Fold Cross Validation**  

<img src="figs/ridge_text_train.png" width=600>

The Gridsearch found the max mean cross validation score with with a regularization parameter (alpha) of ~16.357. Based on the non-zero alpha we can see that the regularization technique is having an impact. The explained variance of the model is not improved in terms of it's generalization to test data. 



### Lasso

Since we have some features that are likely not contributing to the explained variance, A LASSO regression might provide a better result as its regularization parameter can reduce these coefficients to zero. This should reduce the impact of covariance on the model. 

Objective Function fo LASSO:

$ \text{minimize} \left( \frac{1}{2n} \sum_{i=1}^{n} (y_i - \beta_0 - \sum_{j=1}^{p} \beta_j x_{ij})^2 + \lambda \sum_{j=1}^{p} |\beta_j| \right) $

where:

- $(n)$ is the number of samples,
- $(p)$ is the number of features,
- $(y_i)$ is the target value for the $(i)$th sample,
- $(x_{ij})$ is the value of the $(j)$th feature for the $(i)$th sample,
- $(\beta_0, \beta_1, \ldots, \beta_p)$ are the coefficients,
- $(\lambda)$ is the regularization parameter.

**Results:**
```Markdown
LASSO REGRESSION
Best alpha: 0.05898896425508499
Mean Squared Error with best alpha: 0.9471284759470703
R-squared: 0.5824270969636274
```

These results are the same as the multivariate regression, suggesting that that is not a high enough degree of co-linearity for LASSO to penalize any feature enough to reduce the coefficient to zero. 


<img src="figs/lasso_coef.png" width=800>

**Grid Search over Alpha with 5 Fold Cross Validation**  

<img src="figs/lasso_text_train.png" width=600>

The Gridsearch found the max mean cross validation score with with a regularization parameter (alpha) of ~0.0590. 


___
## Significance Testing

<img src="figs/model_significance_testing_p_vals.png" width=800>

Significance Testing was done using p_values of F-Tests of the models against each other and a 'random' model we made based on normal distribution of data. Results we failed to reject there was a significant difference between any model we constructed but they were all significantly better then
an informed random guess.

Would love some commentary on here on if and how we can improve? Finding a null hypothesis for predictions seems difficult?


___
## Results

Our best models returned an explained variance ratio of ~0.58. Due to the complexity of market factors impacting price of Pollock, this seems to be a reasonable result. It is clear that there are many features not included in our data set that are influencing price - as anticipated.  

Based on the regression coefficients, Domestic Pollock Price is most strongly influenced by Domestic Pollock Amount. This relationship makes intuitive sense, since the coefficient is negative and we can assume that increased catches negatively impact market price. 

In terms of the influence of imports on Domestic Pollock Price, the volume of Imported Haddock had a relatively large negative coefficient with a significant (<0.01) p-value. This seems to indicate that there is an inverse relationship between these values and Domestic Pollock price, partially supporting our hypothesis that imported Haddock and Cod from the Barents Sea impact Pollock Prices in Maine. The volume of Imported Cod however, did not seem to exhibit a significant effect on the price of Domestic Pollock. 

___
## Next Steps
Given the inherent complexity of price prediction and the limited nature of our data sample, we found it difficult to make any firm conclusions. Our work is limited by the observational nature of the data, limited data samples, and the complexity of the models employed. Unconsidered disruptors to the fishing industry, market forces and features not included in our approach likely have significant effects on Domestic Pollock Price.

Next steps for the project may include seeking input from domain experts at GMRI; searching for more granular data or implementing additional modeling approaches not yet explored.

___ 
## Acknowledgements
1) Stakeholder meeting with Dr. Kanae Tokunaga, Senior Scientist at GMRI, March 12, 2024.

2) [NOAA 2020 Fisheries of the United States, May 2022](https://media.fisheries.noaa.gov/2022-05/Fisheries-of-the-United-States-2020-Report-FINAL.pdf)

3) [Barents Sea cod quota drops by 20 percent for third straight year](https://www.seafoodsource.com/news/supply-trade/barents-sea-cod-quota-drops-by-20-percent-for-third-straight-year#:~:text=Norway)

4) [Tight cod supplies, better for pollock | GLOBEFISH | Food and Agriculture Organization of the United Nations](https://fao.org/in-action/globefish/market-reports/resource-detail/en/c/1655476/)

5) [Groundfish Forum predicts wild-caught whitefish supplies will remain flat in 2024](https://www.seafoodsource.com/news/supply-trade/groundfish-forum-predicts-wild-caught-whitefish-supplies-remain-flat-in-2024)

6) [Groundfish: Supplies slightly down in 2023 | GLOBEFISH | Food and Agriculture Organization of the United Nations](https://www.fao.org/in-action/globefish/market-reports/resource-detail/en/c/1634023/)

7) [Supplies may become tighter | GLOBEFISH | Food and Agriculture Organization of the United Nations](https://www.fao.org/in-action/globefish/market-reports/resource-detail/en/c/1460139/)

8) [An Introduction to Statistical Learning with Applications in Python](https://www.statlearning.com/resources-python)

9) [Scikit Learn - Supervised Learning](https://scikit-learn.org/stable/supervised_learning.html)
    * [Ordinary Least Squares](https://scikit-learn.org/stable/modules/linear_model.html#ordinary-least-squares)
    * [Ridge Regression and classification](https://scikit-learn.org/stable/modules/linear_model.html#ridge-regression-and-classification)
    * [Lasso](https://scikit-learn.org/stable/modules/linear_model.html#lasso)

10) [Statmodels - Regression and Linear Models](https://www.statsmodels.org/dev/user-guide.html#regression-and-linear-models)


## Special thanks to:
* Dr. Phillip Bogden
* Dr. Kanae Tokunaga
* Edward Wong