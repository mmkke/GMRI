___
## Goal
 

___
## Approach


___
## Prep Data

### Format Data for Modeling


### Select Countries of Interest

```python
 data_filtered = data[data['Country'].isin(['USA', 'ICELAND', 'NORWAY', 'RUSSIAN FEDERATION'])]
 ```

### Create Pivot Tables


### Check Distribution of Features

<img src="figs/value_dist.png" width=800>
<img src="figs/amount_dist.png" width=800>

### Transform Value Features

<img src="figs/amount_log_dist.png" width=800>

### Set Time Frame


### Drop NaN Values and Impute

<img src="figs/missingness1.png" width=800>

<img src="figs/missingness2.png" width=800>

<img src="figs/dist_imputed_values.png" width=400><img src="" width=400>

<img src="figs/missingness3.png" width=800>



### Visualize Data

<img src="figs/data_viz_value.png" width=800>

<img src="figs/data_viz_amount.png" width=800>


### Correlations and Pairplots

<img src="figs/heatmap.png" width=800>

<img src="figs/pairplots.png" width=800>


### Shuffling


### Scaling


### Export Feature Matrix and Target Vector													


___
## Modeling

### Statsmodel ANOVA

<img src="figs/statmodel_summary.png" width=800>

<img src="figs/statsmodel_residuals.png" width=800>

### Univariate Regression

<img src="figs/univariate_Cod_Domestic_Kilos.png" width=400><img src="figs/univariate_Cod_Domestic_USD.png" width=400>
<img src="figs/univariate_Cod_Imported_Kilos.png" width=400><img src="figs/univariate_Cod_Imported_USD.png" width=400>
<img src="figs/univariate_Haddock_Domestic_Kilos.png" width=400><img src="figs/univariate_Haddock_Domestic_USD.png" width=400>
<img src="figs/univariate_Haddock_Imported_Kilos.png" width=400><img src="figs/univariate_Haddock_Imported_USD.png" width=400>
<img src="figs/univariate_Hake_Domestic_Kilos.png" width=400><img src="figs/univariate_Hake_Domestic_USD.png" width=400>
<img src="figs/univariate_Pollock_Domestic_Kilos.png" width=400><img src="figs/univariate_Pollock_Imported_Kilos.png" width=400>
<img src="figs/univariate_Pollock_Imported_USD.png" width=400>
<img src="figs/univariate_r2_values.png" width=800>


### Multivariate Regression

<img src="figs/multi_reg_result.png" width=800>

<img src="figs/mulit_reg_coefs.png" width=800>

<img src="figs/mulit_reg_residuals.png" width=600>


### PCR

<img src="figs/PCR_explained_variance.png" width=800>

<img src="figs/PCR_text_train.png" width=600>


### Ridge

<img src="figs/ridge_coef.png" width=800>

<img src="figs/ridge_text_train.png" width=600>


### Lasso

<img src="figs/lasso_coef.png" width=800>

<img src="figs/lasso_text_train.png" width=600>

<img src="figs/LARS_path.png" width=800>



___
## Selecting the Best Modeling Approach


___
## Signifigance Testing


___
## Results/Conclusion


___ 
## Acknowledgements



