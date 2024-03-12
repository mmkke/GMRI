## Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from eda_utils import *


def main():

    # set params for matplotlib and seaborn
    set_params()

    # get data
    df = get_data()

    # preprocess data
    df= preprocess_data(df)

    #this is just for processing no analysis
    adjustForInflation('data/combined_data_2006-2024.csv', "AvgPrice_per_Kilo","MonthNum", "YearNum")

    ## Visualizations

    # Price distribution by fish type, boxplots
    price_distribution_boxplots(df)
    # Price vs amount sold, by import/domestic and by fish type
    amnt_vs_price_scatterplots(df)
    # Amount solde by species, total and by year
    amnt_sold_by_species_barplots(df)
    # amount sold over time, 2004-2024
    amnt_sold_over_time_lineplot(df)
    # amount sold over time, by species, domestic/imported
    amnt_sold_over_time_by_species_lineplots(df)
    # amount sold over time by country of origin
    amnt_sold_over_time_by_origin_lineplots(df)

if __name__ == "__main__":

    main()