#!/usr/bin/env python3

## Libraries
# general
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# preprocessing
import missingno as msn
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


###################################################################################################################
## Modules

from eda_utils import *

###################################################################################################################
## Functions

###################################################################################################################
## Main

def main():
    
    ## Set RC params
    set_params()

    ## Load Data

    # get data
    data = get_data()

    # preprocess data
    data = preprocess_data(data)
    
    ## Prep Data

    ### Select Countries of Interest
    data_filtered = data[data['Country'].isin(['USA', 'ICELAND', 'NORWAY', 'RUSSIAN FEDERATION'])]

    ### Create Pivot Tables
    # Drop the 'Country' column
    data_without_country = data_filtered.drop('Country', axis=1)

    # Create a unique identifier for each fish group by its import status
    data_without_country['FishGroup_ImportStatusValue'] = np.where(data_without_country['Imported'] == 'Yes',
                                                            data_without_country['FishGroup'] + "_Imported_USD",
                                                            data_without_country['FishGroup'] + "_Domestic_USD")

    # Create a unique identifier for each fish group by its import status
    data_without_country['FishGroup_ImportStatusAmount'] = np.where(data_without_country['Imported'] == 'Yes',
                                                            data_without_country['FishGroup'] + "_Imported_Kilos",
                                                            data_without_country['FishGroup'] + "_Domestic_Kilos")

    # Pivot the table to have dates as rows and the unique fish group import statuses as columns, with average prices as values
    df_value = data_without_country.pivot_table(index='YYYY/MM', 
                                                columns='FishGroup_ImportStatusValue', 
                                                values='AvgPrice_per_Kilo',
                                                aggfunc='mean')
    # Pivot the table to have dates as rows and the unique fish group import statuses as columns, with average prices as values
    df_amount = data_without_country.pivot_table(index='YYYY/MM', 
                                                columns='FishGroup_ImportStatusAmount', 
                                                values='AmountSold_by_Kilo',
                                                aggfunc='sum')


    ### Check Distribution of Features
    ## value
    fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(16, 8))

    for n, column in enumerate(df_value.columns):
        i = n // 5
        j = n % 5
        ax[i, j].hist(df_value[column], bins=10)  
        ax[i, j].set_title(f'{column}')
        ax[i, j].set_xlabel(f'Value')
        ax[i, j].set_ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('figs/value_dist', bbox_inches='tight')
    plt.show()

    ## amount
    fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(16, 8))

    for n, column in enumerate(df_amount.columns):
        i = n // 5
        j = n % 5
        ax[i, j].hist(df_amount[column], bins=10)  
        ax[i, j].set_title(f'{column}')
        ax[i, j].set_xlabel(f'Value')
        ax[i, j].set_ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('figs/amount_dist', bbox_inches='tight')
    plt.show()

    ### Transform Value Features

    log_transform_amount = True
    if log_transform_amount:
        df_amount = df_amount.applymap(lambda x: np.log(x))

        # plot log transformed data
        fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(16, 8))
        for n, column in enumerate(df_amount.columns):
            i = n // 5
            j = n % 5
            ax[i, j].hist(df_amount[column], bins=10)  # Adjust the number of bins as needed
            ax[i, j].set_title(f'{column}')
            ax[i, j].set_xlabel(f'Value')
            ax[i, j].set_ylabel('Frequency')
        plt.tight_layout()
        plt.savefig('figs/amount_log_dist', bbox_inches='tight')
        plt.show()

    ### Join Dataframe
    df_combined = df_value.join(df_amount)
    print(df_combined.head(10))

    ### Set Time Frame
    start_period = pd.Period('2014-01', freq='M')
    end_period = pd.Period('2024-04', freq='M')
    filtered_df_range = df_combined[(df_combined.index >= start_period) & (df_combined.index <= end_period)].copy()

    ### Drop NaN Values and Impute

    # check missingness
    msn.matrix(filtered_df_range)
    plt.tight_layout()
    plt.savefig('figs/missingness1', bbox_inches='tight')
    plt.show()

    # drop columns with worst missingness
    filtered_df_range = filtered_df_range.drop(['Hake_Imported_USD', 'Hake_Imported_Kilos', 'Redfish_Imported_USD', 'Redfish_Domestic_USD', 'Redfish_Domestic_Kilos', 'Redfish_Imported_Kilos'], axis=1)
    msn.matrix(filtered_df_range)
    plt.tight_layout()
    plt.savefig('figs/missingness2', bbox_inches='tight')
    plt.show()

    # plot the dist of columns to impute
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    # import pollock amount
    sns.histplot(data=filtered_df_range, x='Pollock_Imported_Kilos', ax=axs[0])
    axs[0].set_title('Distribution of Pollock_Imported_Kilos')
    # imported pollock value
    sns.histplot(data=filtered_df_range, x='Pollock_Imported_USD', ax=axs[1])
    axs[1].set_title('Distribution of Pollock_Imported_USD')
    plt.tight_layout()
    plt.savefig('figs/dist_imputed_values', bbox_inches='tight')
    plt.show()

    # impute values
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    cols_to_impute = ['Pollock_Imported_Kilos', 'Pollock_Imported_USD']
    filtered_df_range[cols_to_impute] = imp_mean.fit_transform(filtered_df_range[cols_to_impute])

    # drop remaining Nans
    filtered_df_range.dropna(axis=0, inplace=True)
    msn.matrix(filtered_df_range)
    plt.tight_layout()
    plt.savefig('figs/missingness3', bbox_inches='tight')
    plt.show()

    ### Visualize Data
    # amount
    plt.plot(filtered_df_range.index.to_timestamp(), filtered_df_range.Cod_Domestic_USD, marker='o', label='Cod - Domestic')
    plt.plot(filtered_df_range.index.to_timestamp(), filtered_df_range.Cod_Imported_USD, marker='o', color='red', label='Cod - Imported')
    plt.plot(filtered_df_range.index.to_timestamp(), filtered_df_range.Haddock_Domestic_USD, marker='o', color='green', label='Haddock - Domestic')
    plt.plot(filtered_df_range.index.to_timestamp(), filtered_df_range.Haddock_Imported_USD, marker='o', color='yellow', label='Haddock - Imported')
    plt.plot(filtered_df_range.index.to_timestamp(), filtered_df_range.Pollock_Domestic_USD, marker='o', color='orange', label='Polllock - Domestic')
    plt.plot(filtered_df_range.index.to_timestamp(), filtered_df_range.Pollock_Imported_USD, marker='o', color='purple', label='Pollock - Imported')
    plt.legend()
    plt.title('Amount Over Time, by species and import status')
    plt.xlabel('Time')
    plt.ylabel('Price (USD)')
    plt.savefig('figs/data_viz_amount', bbox_inches='tight')
    plt.show();

    # value
    plt.plot(filtered_df_range.index.to_timestamp(), filtered_df_range.Cod_Domestic_Kilos, marker='o', label='Cod - Domestic')
    plt.plot(filtered_df_range.index.to_timestamp(), filtered_df_range.Cod_Imported_Kilos, marker='o', color='red', label='Cod - Imported')
    plt.plot(filtered_df_range.index.to_timestamp(), filtered_df_range.Haddock_Domestic_Kilos, marker='o', color='green', label='Haddock - Domestic')
    plt.plot(filtered_df_range.index.to_timestamp(), filtered_df_range.Haddock_Imported_Kilos, marker='o', color='yellow', label='Haddock - Imported')
    plt.plot(filtered_df_range.index.to_timestamp(), filtered_df_range.Pollock_Domestic_Kilos, marker='o', color='orange', label='Polllock - Domestic')
    plt.plot(filtered_df_range.index.to_timestamp(), filtered_df_range.Pollock_Imported_Kilos, marker='o', color='purple', label='Pollock - Imported')
    plt.legend()
    plt.title('Avg Price Over Time, by species and import status')
    plt.xlabel('Time')
    plt.ylabel('Amount in Kg (Log Transformed)')
    plt.savefig('figs/data_viz_value', bbox_inches='tight')
    plt.show();

    ### Correlation Heatmap and Pairplots

    # heatmap
    sns.heatmap(filtered_df_range.corr(), vmin=-1, vmax=1, annot=True)
    plt.xticks(rotation=45)
    plt.title('Correlation Heatmap')
    plt.savefig('figs/heatmap', bbox_inches='tight')
    plt.show();

    #pairplot
    sns.pairplot(data=filtered_df_range)
    plt.title('Feature Pairplots', bbox_inches='tight')
    plt.show();
    
    ### Shuffle
    shuffled_df = filtered_df_range.sample(frac=1, random_state=42)

    ### Create Feature Matrix and Target Vector

    cols = shuffled_df.columns
    X_cols = cols.drop('Pollock_Domestic_USD')
    X = shuffled_df[X_cols]
    y = shuffled_df[['Pollock_Domestic_USD']]

    ### Scaling

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    ### Save to CSVs
    X_scaled.to_csv('data/feature_matrix.csv', index=False)
    y.to_csv('data/target_vector.csv', index=False)


###################################################################################################################
if __name__ == '__main__':

    main()
