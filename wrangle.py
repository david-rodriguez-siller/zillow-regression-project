import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
import sklearn.preprocessing
import sklearn.feature_selection
from sklearn.feature_selection import SelectKBest, f_regression, RFE

import env


# this function is good

def zillow_data():
    '''
        This function reads the zillow database from the Codeup db into a df.
    '''
    sql_query = """
                SELECT bedroomcnt, bathroomcnt, basementsqft, garagetotalsqft,
                calculatedfinishedsquarefeet, poolcnt, numberofstories,
                taxvaluedollarcnt, yearbuilt, fips 
                FROM properties_2017
                JOIN propertylandusetype USING(propertylandusetypeid)
                JOIN predictions_2017 USING(parcelid)
                WHERE propertylandusetype.propertylandusetypeid = 261 AND 279
                AND predictions_2017.transactiondate LIKE '2017%%';
                """
    
    # Read in DataFrame from Codeup db.
    df = pd.read_sql(sql_query, get_connection('zillow'))
    
    return df

def get_connection(db, user = env.user, host = env.host, password = env.password):
    '''
        This function uses my info from my env file to
        create a connection url to access the Codeup db.
        It takes in a string name of a database as an argument.
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

def wrangle_zillow(df):
    '''
       This function replaces blank spaces with nan values, then those values are dropped. Columns iwth 
       nan values are filled with 0/1 depending on the column. The square feet columns are combined into 
       one column and the individual columns that make up the combined square feet column are dropped. 
       Furthermore, outliers are dropped from the DF along with the choice of dropping rows that have 0 
       bathrooms and less than 200 square feet. Columns are renamed and the fips column is mapped to the 
       meaning of the fips code.
    '''
    
    # replace blank spaces with nan's
    df = df.replace(r'^\s*$', np.nan, regex=True)
    
    # replace nan's with zero on columns that make sense to have 0 square feet or 0 count
    df['basementsqft'] = df['basementsqft'].fillna(0)
    df['garagetotalsqft'] = df['garagetotalsqft'].fillna(0)
    df['poolcnt'] = df['poolcnt'].fillna(0)
    
    # replace nan's on number of stories column with 1
    df['numberofstories'] = df['numberofstories'].fillna(1)
    
    # combine sqrft columns into one and drop unneeded columns
    df['comb_sq_ft'] = df['basementsqft'] + df['garagetotalsqft'] + df['calculatedfinishedsquarefeet']
    df.drop(columns = ['basementsqft', 'garagetotalsqft', 'calculatedfinishedsquarefeet'], inplace = 
            True)
    
    # drop nan's
    df = df.dropna()
    
    # remove outliers
    z_scores = stats.zscore(df)
    abs_z_scores = np.abs(z_scores)
    filtered_entries = (abs_z_scores < 3).all(axis = 1)
    df = df[filtered_entries]
   
    # removing single family houses with no bathrooms
    # 121 values
    df = df[df['bathroomcnt'] != 0]
    
    # removing single family houses with less than 200 sqared feet
    # 2 values
    df = df[df['comb_sq_ft'] > 200]
    
    # rename columns
    df = df.rename(columns = {'bedroomcnt': 'bedroom_cnt', 'bathroomcnt': 'bathroom_cnt', 
                              'taxvaluedollarcnt': 'assessed_tax_value', 'yearbuilt': 'year_built', 
                              'poolcnt': 'pool_cnt', 'numberofstories': 'nbr_stories'})
    
    # map fips codes to locations
    df['location'] = df['fips']
    df['location'] = df['location'].map({6037.0: 'Los Angeles, CA', 6059.0: 'Orange, CA', 6111.0: 
                                         'Ventura, CA'})
    
    return df

def encode_split(df):
    '''
        This function encodes the location column and concatenates with the original df after dropping
        the encoded column. Then the concatenated df is split into train, validate, test df's.
    '''
    encoded_df = pd.get_dummies(df['location'], drop_first = False)
    concat_df = df.drop(columns = ['location'])
    concat_df = pd.concat([concat_df, encoded_df], axis = 1)
    train_and_validate, test = train_test_split(concat_df, random_state = 123)
    train, validate = train_test_split(train_and_validate)
    
    return train, test, validate

def standard_scaler_tvt(train, validate, test):
    # list of columns float and int dtypes
    num_cols = list(train.select_dtypes(include = ['float64', 'int64', 'complex']).columns)
    
    # standard scaler object
    scaler = sklearn.preprocessing.StandardScaler()
    
    # fit scaler
    scaler.fit(train[num_cols])
    
    # scale
    train_scaled = scaler.transform(train[num_cols])
    validate_scaled = scaler.transform(validate[num_cols])
    test_scaled = scaler.transform(test[num_cols])

    # new column names
    new_column_names = [c + '_scaled' for c in num_cols]

    # add column names if scaled df is needed separately
    train_scaled = pd.DataFrame(train_scaled, columns = num_cols)
    validate_scaled = pd.DataFrame(validate_scaled, columns = num_cols)
    test_scaled = pd.DataFrame(test_scaled, columns = num_cols)
    
    return train_scaled, validate_scaled, test_scaled

def show_features_rankings(X_train, rfe):
    """
    Takes in a dataframe and a fit RFE object in order to output the rank of all features
    """
    # rfe here is reference rfe from cell 15
    var_ranks = rfe.ranking_
    var_names = X_train.columns.tolist()
    ranks = pd.DataFrame({'Var': var_names, 'Rank': var_ranks})
    ranks = ranks.sort_values(by = "Rank", ascending = True)
    return ranks
