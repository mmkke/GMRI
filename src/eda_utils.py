## Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



def set_params():
    '''Sets RC params for matplotlib and seaborn plots.'''

    ### Matplotlib

    # Set the font size for titles
    plt.rcParams['axes.titlesize'] = 20

    # Set the font size for labels on the x-axis and y-axis
    plt.rcParams['axes.labelsize'] = 16

    # Set the font size for ticks on the x-axis and y-axis
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12

    # Set the width of lines in plots
    plt.rcParams['lines.linewidth'] = 2.0

    # Set the default figure size
    plt.rcParams['figure.figsize'] = (12, 6)

    # Set the default style to use
    plt.style.use('ggplot')

    # Tick label non scientific
    plt.rcParams['axes.formatter.useoffset'] = False

    # Set the default savefig format
    plt.rcParams['savefig.format'] = 'png'

    ### Seaborn

    # Set the default style
    sns.set_style("darkgrid")

    # Set the default palette
    sns.set_palette("deep")

    # Set the default context
    sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})

    # Set the default font
    sns.set(font='Arial')



def get_data(filename='data/combined_data_2006-2024.csv'):

    # get combined dataframe
    df = pd.read_csv(filename)
    print(df.head())
    return df

def preprocess_data(df):


    # create period col in format YYYY/MM
    df['YYYY/MM'] = pd.to_datetime(df['MonthNum'].astype(str) + df['YearNum'].astype(str), format='%m%Y')
    df['YYYY/MM'] = df['YYYY/MM'].dt.to_period('M')


    ## combine entries where FIshGroup and countries are the same within a given month

    # set index to dt col
    df_dt_idx = df.set_index('YYYY/MM', inplace=False)

    df_dt_idx['TotalSoldAtPrice'] = df_dt_idx['AmountSold_by_Kilo'] * df_dt_idx['AvgPrice_per_Kilo']

    #group by dt, fishgroup, and country and aggregate, then reset index
    df_agg = df_dt_idx.groupby(['YYYY/MM', 'FishGroup', 'Country']).agg(
        AmountSold_by_Kilo=('AmountSold_by_Kilo', 'sum'),
        TotalSoldAtPrice=('TotalSoldAtPrice', 'sum')
    ).reset_index()

    # Calculate the weighted average price per kilo for each group
    df_agg['AvgPrice_per_Kilo'] = df_agg['TotalSoldAtPrice'] / df_agg['AmountSold_by_Kilo']

    # Drop the 'TotalSoldAtPrice' column if you no longer need it
    df_agg.drop('TotalSoldAtPrice', axis=1, inplace=True)


    # add column for imported vs domestic

    df_agg['Imported'] = np.where(df_agg['Country'] != 'USA', 'Yes', 'No')

    # drop outliers
    print("OUTLIERS:")
    print(df_agg[df_agg['AvgPrice_per_Kilo'] > 40])
    df_agg = df_agg[df_agg['AvgPrice_per_Kilo'] <= 40]


    return df_agg


def price_distribution_boxplots(df):

    fig, axs = plt.subplots(1, 2, figsize=(18, 9))

    sns.boxplot(data=df, x='FishGroup', y='AvgPrice_per_Kilo', ax=axs[0])
    axs[0].set_xlabel('Fish Species')
    axs[0].set_ylabel('Average Price')
    axs[0].set_title('Distribution of Prices per Species')
    axs[0].ticklabel_format(style='plain', axis='y')


    sns.boxplot(data=df, x='FishGroup', y='AvgPrice_per_Kilo', hue='Imported', dodge=True, ax=axs[1])
    axs[1].set_xlabel('Fish Species')
    axs[1].set_ylabel('Average Price')
    axs[1].set_title('Distribution of Prices per Species')
    axs[1].ticklabel_format(style='plain', axis='y')



    plt.tight_layout()
    plt.savefig('figs/price_distribution_boxplots.png', bbox_inches='tight')
    plt.show()

def price_distribution_boxplots2(df):

    # Extract the year from the Period column
    df['Year'] = df['YYYY/MM'].dt.year

    fig, axs = plt.subplots(2, 1, figsize=(18, 9))

    sns.boxplot(data=df, x='Year', y='AvgPrice_per_Kilo', hue='FishGroup', dodge=True, ax=axs[0])
    axs[0].set_xlabel('Year')
    axs[0].set_ylabel('Average Price per Kg')
    axs[0].set_title('Distribution of Prices per Species')
    axs[0].ticklabel_format(style='plain', axis='y')


    sns.boxplot(data=df, x='Year', y='AvgPrice_per_Kilo', hue='Imported', dodge=True, ax=axs[1])
    axs[1].set_xlabel('Year')
    axs[1].set_ylabel('Average Price per Kg')
    axs[1].set_title('Distribution of Prices per Species')
    axs[1].ticklabel_format(style='plain', axis='y')


    plt.tight_layout()
    plt.savefig('figs/price_distribution_boxplots2.png', bbox_inches='tight')
    plt.show()

def amnt_vs_price_scatterplots(df):

    fig, ax = plt.subplots(1, 2, figsize=(18,9))

    sns.scatterplot(ax=ax[0], data=df, x='AmountSold_by_Kilo', y='AvgPrice_per_Kilo', hue='Imported')
    ax[0].set_xlabel(r'Amount Sold in kg ($\log_{10}$)')
    ax[0].set_xscale('log')
    ax[0].set_ylabel('Price')
    ax[0].set_title('Amount Sold vs Price, origin')
    #ax[0].ticklabel_format(style='plain', axis='both')


    sns.scatterplot(ax=ax[1], data=df, x='AmountSold_by_Kilo', y='AvgPrice_per_Kilo', hue='FishGroup', alpha=0.8)
    ax[1].set_xlabel(r'Amount Sold in kg ($\log_{10}$)')
    ax[1].set_xscale('log')
    ax[1].set_ylabel('Price')
    ax[1].set_title('Amount Sold vs Price, species')
    #ax[1].ticklabel_format(style='plain', axis='both')


    plt.tight_layout()
    plt.savefig('figs/amnt_vs_price_scatterplots.png', bbox_inches='tight')
    plt.show()

def price_by_species_lineplots(df):

    # Extract the year from the Period column
    df['Year'] = df['YYYY/MM'].dt.year

    fig, axs = plt.subplots(1, 2, figsize=(16, 6))

    # plot cumulative distribution
    sns.lineplot(data=df, x='Year', y='AvgPrice_per_Kilo', hue='Imported', ax=axs[0])
    axs[0].set_xlabel('Year')
    axs[0].set_ylabel('Price in USD per kg')
    axs[0].set_title('Average Price in USD per kg, by origin')
    axs[0].tick_params(axis='x', rotation=45)
    axs[0].ticklabel_format(style='plain', axis='y')
    axs[0].set_xticks(np.arange(2004, 2025, 1))

    # plot dist per year
    sns.lineplot(data=df, x='Year', y='AvgPrice_per_Kilo', hue='FishGroup', ax=axs[1])
    axs[1].set_xlabel('Year')
    axs[1].set_ylabel('Price in USD per kg')
    axs[1].set_title('Average Price in USD per kg, by species')
    axs[1].tick_params(axis='x', rotation=45)
    axs[1].ticklabel_format(style='plain', axis='y')
    axs[0].set_xticks(np.arange(2004, 2025, 1))

    plt.tight_layout()
    plt.savefig('figs/price_over_time_lineplots.png', bbox_inches='tight')
    plt.show()

def amnt_sold_by_species_barplots(df):

    # Extract the year from the Period column
    df['Year'] = df['YYYY/MM'].dt.year

    fig, axs = plt.subplots(1, 2, figsize=(16, 6))

    # plot cumulative distribution
    sns.barplot(data=df, x='FishGroup', y='AmountSold_by_Kilo', hue='Imported', ax=axs[0])
    axs[0].set_xlabel('Fish Species')
    axs[0].set_ylabel('Amount Sold (kg)')
    axs[0].set_title('Total Amount Sold, by species')
    axs[0].tick_params(axis='x', rotation=45)
    axs[0].ticklabel_format(style='plain', axis='y')


    # plot dist per year
    sns.barplot(data=df, x='Year', y='AmountSold_by_Kilo', hue='FishGroup', ax=axs[1], 
                dodge=True, estimator='sum', errorbar=None)
    axs[1].set_xlabel('Year')
    axs[1].set_ylabel('Amount Sold (kg)')
    axs[1].set_title('Total Amount Sold, by species')
    axs[1].tick_params(axis='x', rotation=45)
    axs[1].ticklabel_format(style='plain', axis='y')


    plt.tight_layout()
    plt.savefig('figs/amnt_sold_by_species_barplot.png', bbox_inches='tight')
    plt.show()


def amnt_sold_over_time_lineplot(df):


    sns.lineplot(data=df, x=df['YYYY/MM'].apply(lambda x: x.to_timestamp()), y='AmountSold_by_Kilo', hue='FishGroup', errorbar=None)
    plt.xlabel('Time')
    plt.ylabel('Amount Sold (kg)')
    plt.title('Fish Sold over Time, by species')
    plt.tight_layout()
    plt.savefig('figs/amnt_sold_over_time_lineplot.png', bbox_inches='tight')
    plt.show()


def amnt_sold_over_time_by_species_lineplots(df):

    fishes = df['FishGroup'].unique()

    fig, ax = plt.subplots(5, 2, figsize=(18,18))

    for idx, fish in enumerate(fishes):

        sns.lineplot(ax = ax[idx,0], data=df[df['FishGroup'] == fish], x=df['YYYY/MM'].apply(lambda x: x.to_timestamp()), y='AmountSold_by_Kilo')
        ax[idx,0].set_xlabel('Time')
        ax[idx,0].set_ylabel('Amount Sold (kg)')
        ax[idx,0].set_title(f'{fish} Sold over Time, by species')
        ax[idx,0].ticklabel_format(style='plain', axis='y')
        

        sns.lineplot(ax = ax[idx,1], data=df[df['FishGroup'] == fish], x=df['YYYY/MM'].apply(lambda x: x.to_timestamp()), y='AmountSold_by_Kilo', hue='Imported')
        ax[idx,1].set_xlabel('Time')
        ax[idx,1].set_ylabel('Amount Sold (kg)')
        ax[idx,1].set_title(f'{fish} Sold over Time, by species')
        ax[idx,1].ticklabel_format(style='plain', axis='y')
        

    plt.tight_layout()
    plt.savefig('figs/amnt_sold_over_time_by_species_lineplots.png', bbox_inches='tight')
    plt.show()


def price_over_time_by_species_lineplots(df):

    fishes = df['FishGroup'].unique()

    fig, ax = plt.subplots(5, 2, figsize=(18,18))

    for idx, fish in enumerate(fishes):

        sns.lineplot(ax = ax[idx,0], data=df[df['FishGroup'] == fish], x=df['YYYY/MM'].apply(lambda x: x.to_timestamp()), y='AvgPrice_per_Kilo')
        ax[idx,0].set_xlabel('Time')
        ax[idx,0].set_ylabel('Average Price per Kg')
        ax[idx,0].set_title(f'{fish} Price over Time, by species')
        ax[idx,0].ticklabel_format(style='plain', axis='y')
        

        sns.lineplot(ax = ax[idx,1], data=df[df['FishGroup'] == fish], x=df['YYYY/MM'].apply(lambda x: x.to_timestamp()), y='AvgPrice_per_Kilo', hue='Imported')
        ax[idx,1].set_xlabel('Time')
        ax[idx,1].set_ylabel('Average Price per Kg')
        ax[idx,1].set_title(f'{fish} Price over Time, by species')
        ax[idx,1].ticklabel_format(style='plain', axis='y')
        

    plt.tight_layout()
    plt.savefig('figs/price_over_time_by_species_lineplots.png', bbox_inches='tight')
    plt.show()


def amnt_sold_over_time_by_origin_lineplots(df):


    fig, axs = plt.subplots(3, 1, figsize=(10, 18))

    sns.lineplot(data=df, x=df['YYYY/MM'].apply(lambda x: x.to_timestamp()), y='AmountSold_by_Kilo', hue='Imported', ax=axs[0])
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Amount Sold (kg)')
    axs[0].set_title('Fish Sold over Time, by origin')
    axs[0].ticklabel_format(style='plain', axis='y')

    sns.lineplot(data=df[df['FishGroup'] == 'Cod'], x=df['YYYY/MM'].apply(lambda x: x.to_timestamp()), y='AmountSold_by_Kilo', hue='Country', ax=axs[1])
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Amount Sold (kg)')
    axs[1].set_title('Cod Sold over Time, by origin')
    axs[2].ticklabel_format(style='plain', axis='y')

    sns.lineplot(data=df[df['FishGroup'] == 'Hake'], x=df['YYYY/MM'].apply(lambda x: x.to_timestamp()), y='AmountSold_by_Kilo', hue='Country', ax=axs[2])
    axs[2].set_xlabel('Time')
    axs[2].set_ylabel('Amount Sold (kg)')
    axs[2].set_title('Hake Sold over Time, by origin')
    axs[2].ticklabel_format(style='plain', axis='y')

    plt.tight_layout()
    plt.savefig('figs/amnt_sold_over_time_by_origin_lineplots.png', bbox_inches='tight')
    plt.show()


def adjustForInflation(filePath, columnToAdjust, monthCol, yearCol):
    inflationData = pd.read_csv('data/BLS_CPI_inflationData_2004_2024.csv')
    df = pd.read_csv(filePath)
    print(inflationData)
   

    df_inflation = (pd.merge(df, inflationData[['scale', 'year', 'period']], left_on=[yearCol, monthCol], right_on=['year', 'period'], how='left')).drop(columns=["period", "year"])
    df_inflation = df_inflation.dropna(subset=['scale'])
    df_inflation['inflationAdjusted_'+columnToAdjust] = df_inflation['scale'] * df_inflation[columnToAdjust]

    df_inflation = df_inflation.sort_values(by=[yearCol, monthCol])
    df_inflation=df_inflation.drop(columns=['scale'])

    filepath = "data/inflation_adjusted_combined_data_2004-2024.csv"
    df_inflation.to_csv(filepath, index=False) 
    print(f'Combined Inflation adjusted data saved at: {filepath}')

    return df_inflation.sort_values(by=[yearCol, monthCol])
