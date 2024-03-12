import numpy as np
import pandas as pd
import requests
import json


def get_noaa_data(api_key='klHRkURJrqCrXMZEtjPoSWwckiYazQFS', url='https://apps-st.fisheries.noaa.gov/ods/foss/trade_data/', query=None, filepath='data/noaa_2006-2024.csv'):
    '''Access NOAA Fisheries API to retrieve data and save to CSV. Data is returned as pandas dataframe.
    Params:
        api_key: klHRkURJrqCrXMZEtjPoSWwckiYazQFS
        url: https://apps-st.fisheries.noaa.gov/ods/foss/trade_data/
        query:{
            "year":{"$gt": 2006}, -> Data from 2006 till present. 
            "name":{"$like":"%GROUNDFISH%"},  -> Only return records for species considered within category: GROUNDFISH
            "custom_district_name":{"$like":"%PORTLAND%"}, ->Only return records of imports to PORTLAND
            "continent":{"$like":"%EU%"} -> Only return records of fish imported from EU
            }
        filepath: data/noaa_2006-2024.csv
    Return:
        dataframe
    '''
    # define params for api query
    api_key = api_key
    url = url

    # no specific query given as arg, proceed with default query
    if query == None:
        query = '''{
            "year":{"$gt": 2004},
            "name":{"$like":"%GROUNDFISH%"}, 
            "custom_district_name":{"$like":"%PORTLAND%"}, 
            "continent":{"$like":"%EU%"}
        }'''

    params = {
        'api_key': api_key,
        'q': query,
        'limit': 9999, 
        'offset': 0,
    }

    response = requests.get(url, params=params)

    # check connection to api
    if response.status_code == 200:
        data = response.json()
        print('Data retrieved:')
        print(data)
        print('Success.')
    else:
        print('Failed to fetch data:', response.status_code, response.text)

    # convert json to pandas dataframe
    df = pd.json_normalize(data['items'])

    # add datetime column to dataframe 
    df['date'] = pd.to_datetime(df['year'].astype(str) + '-' + df['month'].astype(str), format='%Y-%m')

    # save to csv and print filepath
    df.to_csv(filepath, index=False)
    print(f'Data saved to: {filepath}')

    # return dataframe
    return df

def map_fish_group(value):
    '''Mapping function for aggregating fish species.'''

    value = value.lower()  # Convert the string to lowercase for consistent comparison
    if 'cod' in value:
        return 'Cod'
    elif 'haddock' in value:
        return 'Haddock'
    elif 'pollock' in value:
        return 'Pollock'
    elif 'hake' in value or 'whiting' in value:  # 'whiting' maps to 'hake'
        return 'Hake'
    elif 'perch' in value:  # 'perch' maps to 'redfish'
        return 'Redfish'
    else:
        return 'Other'  # Assign 'Other' if none of the conditions above are met  


def create_fishgroup_col():
    '''Filter and aggregate species in PFEX data'''

    # get data
    filepath = "data/pfex_2004-2024.csv"
    df = pd.read_csv(filepath)

    # List of keywords to search for in the 'FishDesc' column, case-insensitive
    keywords = ['cod', 'haddock', 'pollock', 'flounder', 'hake', 'redfish', 'halibut', 'yellowtail']

    # Creating a regex pattern to match any of the keywords
    pattern = '|'.join(keywords)

    # Filter the DataFrame to keep records containing any of the keywords in 'FishDesc'
    groundfish_df = df[df['FishDesc'].str.contains(pattern, case=False, na=False)].copy()

    # Group by 'FishCode' and 'FishDesc', then aggregate to keep the first description
    aggregated = groundfish_df.groupby('FishCode')['FishDesc'].agg(lambda x: x.iloc[0]).reset_index()

    # Define the mapping of starting sequences to FishGroup categories
    fish_group_mapping = {
        'C': 'Cod',
        'FA': 'Halibut',
        'FY': 'Yellowtail',
        'HD': 'Haddock',
        'HW': 'Hake',
        'PA': 'Pollock',
        'PO': 'Redfish',
        'Y': 'Yellowtail'
    }

    # Function to determine the FishGroup based on the FishCode
    def assign_fish_group(fish_code):
        for start_seq, group in fish_group_mapping.items():
            if fish_code.startswith(start_seq):
                return group
        return 'OTHER'  # Default assignment if no match found

    # Apply the function to create the new 'FishGroup' column
    groundfish_df.loc[:, 'FishGroup'] = groundfish_df['FishCode'].apply(assign_fish_group)


    # save processed data as csv
    filepath = "data/pfex_2004-2024_processed.csv"
    groundfish_df.to_csv(filepath, index=False) 
    print(f'Processed data saved at: {filepath}')

def merge_data():
    '''Merges data from NOAA and PFEX into a single dataframe.'''

    ## Load Data

    # get pfex (local) data
    filepath = "data/pfex_2004-2024_processed.csv"
    local_df = pd.read_csv(filepath)

    # get noaa (import) data
    filepath = "data/noaa_2006-2024.csv"
    imports_df = pd.read_csv(filepath)


    ## Local Data

    # Filter the dataframe to remove rows where 'FishGroup' is 'Halibut', 'Yellowtail', or 'OTHER'
    local_df = local_df[~local_df['FishGroup'].isin(['Halibut', 'Yellowtail', 'OTHER'])]

    # Convert 'Sold' from pounds to kilos
    local_df['AmountSold_by_Kilo'] = local_df['Sold'] / 2.20462

    # Convert AvgPrice from $/lb to $/kilo
    local_df['AvgPrice_per_Kilo'] = local_df['AvgPrice'] * 2.20462

    local_df['Country'] = 'USA'

    # filter columns
    local_df = local_df[['AmountSold_by_Kilo', 'AvgPrice_per_Kilo', 'YearNum', 'MonthNum', 'FishGroup', 'Country']]


    ## Import Data

    # filter columns
    imports_df = imports_df[['kilos', 'val', 'year', 'month', 'fus_group1', 'fus_group2', 'cntry_name']]

    # create price per kilo column
    imports_df.loc[:, 'Price'] = imports_df['val'] / imports_df['kilos']

    # Apply the mapping function to the 'fus_group2' column to create the new 'FishGroup' column
    imports_df['FishGroup'] = imports_df['fus_group2'].apply(map_fish_group)

    # Filter out the rows where 'FishGroup' is 'Other'
    imports_df = imports_df[imports_df['FishGroup'] != 'Other']

    # rename cols
    imports_df.rename(columns={
                                'kilos': 'AmountSold_by_Kilo',
                                'Price': 'AvgPrice_per_Kilo',
                                'year': 'YearNum',
                                'month': 'MonthNum',
                                'cntry_name': 'Country'
                            }, inplace=True)
    
    imports_df['AmountSold_by_Kilo'] = imports_df['AmountSold_by_Kilo'].astype(float)


    ## Merge Data

    columns_to_keep = ['AmountSold_by_Kilo', 'AvgPrice_per_Kilo', 'YearNum', 'MonthNum', 'FishGroup', 'Country']

    local_df = local_df[columns_to_keep]
    imports_df = imports_df[columns_to_keep]

    # Concatenate the dataframes
    combined_df = pd.concat([local_df, imports_df], ignore_index=True)  

    filepath = "data/combined_data_2006-2024.csv"
    combined_df.to_csv(filepath, index=False) 
    print(f'Combined data saved at: {filepath}')

    return combined_df




def inflationDataApiCall(startYear, EndYear):
    #adapted from https://www.bls.gov/developers/api_python.htm
    #data from
    #https://data.bls.gov/timeseries/CUUR0000SA0 Consumer Price Index for All Urban Consumers (CPI-U)

    
    headers = {'Content-type': 'application/json'}
    data = json.dumps({"seriesid": ['CUUR0000SA0'],"startyear":str(startYear), "endyear":str(EndYear)})
    p = requests.post('https://api.bls.gov/publicAPI/v2/timeseries/data/', data=data, headers=headers)
    json_data = json.loads(p.text)

    cpiData = pd.DataFrame(json_data['Results']['series'][0]['data'])
    cpiData['period'] = cpiData['period'].str.replace(r'^M0|^M', '', regex=True)

    return cpiData[["year","period","value"]]

def getInflationData():
    #get data for 10 year periods

    #hardcoded years to what we're working with and when data is available
    #only seems to be able to call a decade at a time using their limited api
    cpi1= inflationDataApiCall(2004,2014)
    cpi2 =  inflationDataApiCall(2014,2024)

    #concat data
    cpiData= pd.concat([cpi1,cpi2], ignore_index=True)
    #sort by year and period
    cpiData = cpiData.sort_values(by=['year', 'period'])

    #changed to nums to allow math operations
    cpiData['value'] = cpiData['value'].astype(float)
    cpiData['year'] = cpiData['year'].astype(int)
    cpiData['period'] = cpiData['period'].astype(int)

    #set base value to the lastest available data
    baseVal = cpiData[cpiData["year"] == cpiData['year'].max()]
    baseVal = baseVal[baseVal["period"] == baseVal['period'].max()]['value']

    #divide the base value by the value of any given month to get the scale
    cpiData['scale'] =  baseVal.iloc[0] / cpiData['value'] 
    cpiData =cpiData.sort_values(by=['year', 'period'])
    
    

    '''
    calculate multiplier for inflation
    Checked math using
    https://data.bls.gov/cgi-bin/cpicalc.pl?cost1=1.65&year1=202312&year2=200401
    '''
    baseVal = cpiData[cpiData["year"] == cpiData['year'].max()]
    baseVal = baseVal[baseVal["period"] == baseVal['period'].max()]['value']
    cpiData['scale'] =  baseVal.iloc[0] / cpiData['value'] 

    filepath = 'data/BLS_CPI_inflationData_2004_2024.csv'
    cpiData.to_csv(filepath, index=False)
    print(f'Inflation Data saved at: {filepath}')

    return cpiData


def getPortFishex():
    '''
    Function to get and save Portland Fish Exchange data
    disclaimer Portland Fish Exchange has no public facing API. However they have no disclaimers about scraping on their site and this is
    publically available data.

    I got this link by observing what requests were made when I downloaded a report from the site.

    Because this is unofficial I"m putting it in a try except block and not removing the df from the data folder with clean

    Please advise
    '''
    
    pfex_2004_2024 = pd.read_csv('https://reports.pfex.org/customreport_csv.asp?submitted=true&startdate=1%2F1%2F2004&enddate=1%2F1%2F2024')
    pfex_2004_2024.to_csv('data/pfex_2004-2024.csv',index=False)
   
    


