import pandas as pd
import numpy as np
from env import get_db_url
from sklearn.model_selection import train_test_split



def prep_iris(iris):
    iris = iris.drop(columns=['species_id','measurement_id'])
    iris = iris.rename(columns={'species_name':'species'})
    dummy_df = pd.get_dummies(iris[['species']], dummy_na=False, drop_first=[True])
    iris = pd.concat([iris, dummy_df], axis = 1)
    iris = iris.drop(columns=['species'])

    return iris

def prep_titantic(df):
    df = df.drop(columns=['passenger_id','embarked','deck','class','age'])
    df.embark_town = df.embark_town.fillna('Southampton')
    dummy_df = pd.get_dummies(df[['sex','embark_town']], dummy_na=False, drop_first=True)
    df = pd.concat([df,dummy_df], axis = 1)
    
    return df.drop(columns=['sex','embark_town'])

def prep_telco(df):
    # replace whitespace only cells with nan
    df = df.replace(" ",np.nan)
    # Drop the rows with NAs 
    df = df.dropna()
    # Change total_charges type to float
    df.total_charges = df.total_charges.astype('float64')
    # Drop unnecessary foreign key ids
    df = df.drop(columns=['payment_type_id','internet_service_type_id','contract_type_id'])
    # Determine the categorical variables - here defined as object data type (non-numeric) and with fewer than 5 values
    catcol = df.columns[(df.nunique()<5)&(df.dtypes == 'object')]
    # Encode categoricals
    dummy_df = pd.get_dummies(df[catcol], dummy_na=False, drop_first=True)
    # Concatenate dummy df to original df
    df = pd.concat([df,dummy_df],axis=1)
    # Remove the original categorical columns after encoding
    df = df.drop(columns=catcol)
    
    return df

def train_validate_test_split(df, target, seed=123):
    '''
    This function takes in a dataframe, the name of the target variable
    (for stratification purposes), and an integer for a setting a seed
    and splits the data into train, validate and test. 
    Test is 20% of the original dataset, validate is .30*.80= 24% of the 
    original dataset, and train is .70*.80= 56% of the original dataset. 
    The function returns, in this order, train, validate and test dataframes. 
    '''
    train_validate, test = train_test_split(df, test_size=0.2, 
                                            random_state=seed, 
                                            stratify=df[target])
    train, validate = train_test_split(train_validate, test_size=0.3, 
                                       random_state=seed,
                                       stratify=train_validate[target])
    return train, validate, test

