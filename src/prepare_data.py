## Libraries

import numpy as np
import pandas as pd
from data_utils import *



def main():
    getPortFishex()
    
    get_noaa_data()
    

    create_fishgroup_col()

    df = merge_data()

    print(df.info(10))

    getInflationData()

    


if __name__ == "__main__":

    main()