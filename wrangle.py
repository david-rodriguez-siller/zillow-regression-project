import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import env


def zillow_data():
    '''
    This function reads the iris data from the Codeup db into a df.
    '''
    sql_query = """
                SELECT bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt,
                yearbuilt, taxamount, fips 
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
       This function replaces blank spaces with nan values, then those values are dropped. Finally,              outliers and properties with 0
       bathrooms and less than 200 square feet are dropped.
    '''
    
    # replace blank spaces with nan's
    df = df.replace(r'^\s*$', np.nan, regex=True)
    
    # drop nan's
    df = df.dropna()
    
    # remove outliers
    z_scores = stats.zscore(df)
    abs_z_scores = np.abs(z_scores)
    filtered_entries = (abs_z_scores < 3).all(axis = 1)
    df = df[filtered_entries]
    df = df[df['bathroomcnt'] != 0]
    df = df[df['calculatedfinishedsquarefeet'] > 200]
    
    return df

def minmax_scaler_tvt(train, validate, test):
    # list of columns float and int dtypes
    num_cols = list(train.select_dtypes(include = ['float64', 'int64', 'complex']).columns)
    
    # min-max scaler object
    scaler = sklearn.preprocessing.MinMaxScaler()
    
    # fit scaler
    scaler.fit(train[num_cols])
    
    # scale
    train_scaled = scaler.transform(train[num_cols])
    validate_scaled = scaler.transform(validate[num_cols])
    test_scaled = scaler.transform(test[num_cols])

   # new column names
    new_column_names = [c + '_scaled' for c in num_cols]

    # add scaled columns to input dataset
    train[new_column_names] = scaler.transform(train[num_cols])
    validate[new_column_names] = scaler.transform(train[num_cols])
    test[new_column_names] = scaler.transform(train[num_cols])
    
    return train, validate, test

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

    # add scaled columns to input dataset
    train[new_column_names] = scaler.transform(train[num_cols])
    validate[new_column_names] = scaler.transform(train[num_cols])
    test[new_column_names] = scaler.transform(train[num_cols])
    
    return train, validate, test

def robust_scaler_tvt(train, validate, test):
    # list of columns float and int dtypes
    num_cols = list(train.select_dtypes(include = ['float64', 'int64', 'complex']).columns)
    
    # robust scaler object
    scaler = sklearn.preprocessing.RobustScaler()
    
    # fit scaler
    scaler.fit(train[num_cols])
    
    # scale
    train_scaled = scaler.transform(train[num_cols])
    validate_scaled = scaler.transform(validate[num_cols])
    test_scaled = scaler.transform(test[num_cols])

    # new column names
    new_column_names = [c + '_scaled' for c in num_cols]

    # add scaled columns to input dataset
    train[new_column_names] = scaler.transform(train[num_cols])
    validate[new_column_names] = scaler.transform(train[num_cols])
    test[new_column_names] = scaler.transform(train[num_cols])
    
    return train, validate, test

def nonlinear_scaler_tvt(train, validate, test):
    # list of columns float and int dtypes
    num_cols = list(train.select_dtypes(include = ['float64', 'int64', 'complex']).columns)
    
    # non-linear scaler object
    scaler = sklearn.preprocessing.QuantileTransformer(output_distribution = 'normal')
    
    # fit scaler
    scaler.fit(train[num_cols])
    
    # scale
    train_scaled = scaler.transform(train[num_cols])
    validate_scaled = scaler.transform(validate[num_cols])
    test_scaled = scaler.transform(test[num_cols])
    
    # new column names
    new_column_names = [c + '_scaled' for c in num_cols]

    # add scaled columns to input dataset
    train[new_column_names] = scaler.transform(train[num_cols])
    validate[new_column_names] = scaler.transform(train[num_cols])
    test[new_column_names] = scaler.transform(train[num_cols])

    return train, validate, test