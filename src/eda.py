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


    ## Visualizations


    price_distribution_boxplots(df)

    amnt_vs_price_scatterplots(df)

    amnt_sold_by_species_barplots(df)

    amnt_sold_over_time_lineplot(df)

    amnt_sold_over_time_by_species_lineplots(df)

    amnt_sold_over_time_by_origin_lineplots(df)

if __name__ == "__main__":

    main()