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

    # group by dt, fishgroup, and country and aggregate, then reset index
    df_agg = df_dt_idx.groupby(['YYYY/MM', 'FishGroup', 'Country']).agg(
        AmountSold_by_Kilo=('AmountSold_by_Kilo', 'sum'),
        AvgPrice_per_Kilo=('AvgPrice_per_Kilo', lambda x: (x * df_dt_idx.loc[x.index, 'AmountSold_by_Kilo']).sum() / x.sum())
    ).reset_index()

    # check that prices were averaged correctly
    df_grouped_fish = df.groupby('FishGroup')[['AvgPrice_per_Kilo', 'AmountSold_by_Kilo']].mean()
    df_agg_grouped_fish = df_agg.groupby('FishGroup')[['AvgPrice_per_Kilo', 'AmountSold_by_Kilo']].mean()
    assert(df_agg_grouped_fish['AvgPrice_per_Kilo'].values.all() == df_grouped_fish['AvgPrice_per_Kilo'].values.all())


    # add column for imported vs domestic

    df_agg['Imported'] = np.where(df_agg['Country'] != 'USA', 'Yes', 'No')

    return df_agg


def price_distribution_boxplots(df):

    fig, axs = plt.subplots(1, 2, figsize=(18, 9))

    sns.boxplot(data=df, x='FishGroup', y='AvgPrice_per_Kilo', ax=axs[0])
    axs[0].set_xlabel('Fish Species')
    axs[0].set_ylabel('Average Price')
    axs[0].set_title('Distribution of Prices per Species')

    sns.boxplot(data=df, x='FishGroup', y='AvgPrice_per_Kilo', hue='Imported', dodge=True, ax=axs[1])
    axs[1].set_xlabel('Fish Species')
    axs[1].set_ylabel('Average Price')
    axs[1].set_title('Distribution of Prices per Species')

    plt.tight_layout()
    plt.savefig('figs/price_distribution_boxplots.png', bbox_inches='tight')
    plt.show()


def amnt_vs_price_scatterplots(df):

    fig, ax = plt.subplots(1, 2, figsize=(18,9))

    sns.scatterplot(ax=ax[0], data=df, x='AmountSold_by_Kilo', y='AvgPrice_per_Kilo', hue='Imported')
    ax[0].set_xlabel('Amount Sold (kg)')
    ax[0].set_ylabel('Price')
    ax[0].set_title('Amount Sold vs Price, origin')

    sns.scatterplot(ax=ax[1], data=df, x='AmountSold_by_Kilo', y='AvgPrice_per_Kilo', hue='FishGroup', alpha=0.8)
    ax[0].set_xlabel('Amount Sold (kg)')
    ax[0].set_ylabel('Price')
    ax[0].set_title('Amount Sold vs Price, species')


    plt.tight_layout()
    plt.savefig('figs/amnt_vs_price_scatterplots.png', bbox_inches='tight')
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

    # plot dist per year
    sns.barplot(data=df, x='Year', y='AmountSold_by_Kilo', hue='FishGroup', ax=axs[1])
    axs[1].set_xlabel('Year')
    axs[1].set_ylabel('Amount Sold (kg)')
    axs[1].set_title('Total Amount Sold, by species')
    axs[1].tick_params(axis='x', rotation=45)

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

        sns.lineplot(ax = ax[idx,1], data=df[df['FishGroup'] == fish], x=df['YYYY/MM'].apply(lambda x: x.to_timestamp()), y='AmountSold_by_Kilo', hue='Imported')
        ax[idx,1].set_xlabel('Time')
        ax[idx,1].set_ylabel('Amount Sold (kg)')
        ax[idx,1].set_title(f'{fish} Sold over Time, by species')

    plt.tight_layout()
    plt.savefig('figs/amnt_sold_over_time_by_species_lineplots.png', bbox_inches='tight')
    plt.show()


def amnt_sold_over_time_by_origin_lineplots(df):


    fig, axs = plt.subplots(3, 1, figsize=(10, 18))

    sns.lineplot(data=df, x=df['YYYY/MM'].apply(lambda x: x.to_timestamp()), y='AmountSold_by_Kilo', hue='Imported', ax=axs[0])
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Amount Sold (kg)')
    axs[0].set_title('Fish Sold over Time, by origin')

    sns.lineplot(data=df[df['FishGroup'] == 'Cod'], x=df['YYYY/MM'].apply(lambda x: x.to_timestamp()), y='AmountSold_by_Kilo', hue='Country', ax=axs[1])
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Amount Sold (kg)')
    axs[1].set_title('Cod Sold over Time, by origin')

    sns.lineplot(data=df[df['FishGroup'] == 'Hake'], x=df['YYYY/MM'].apply(lambda x: x.to_timestamp()), y='AmountSold_by_Kilo', hue='Country', ax=axs[2])
    axs[2].set_xlabel('Time')
    axs[2].set_ylabel('Amount Sold (kg)')
    axs[2].set_title('Hake Sold over Time, by origin')

    plt.tight_layout()
    plt.savefig('figs/amnt_sold_over_time_by_origin_lineplots.png', bbox_inches='tight')
    plt.show()
