## Libraries

import numpy as np
import pandas as pd
from data_utils import *



def main():
    getPortFishex()
    
    get_noaa_data()

    create_fishgroup_col()

    getInflationData()

    df = merge_data(inflation=True)

    print(df.info(10))

if __name__ == "__main__":

    main()