## Libraries

import numpy as np
import pandas as pd
from data_utils import *



def main():

    get_noaa_data()

    create_fishgroup_col()

    df = merge_data()

    print(df.info(10))

    getInflationData()

    adjustForInflation('data/combined_data_2004-2024.csv', "AvgPrice_per_Kilo","MonthNum", "YearNum")


if __name__ == "__main__":

    main()