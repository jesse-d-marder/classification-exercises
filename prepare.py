import pandas as pd
from env import get_db_url


def prep_iris(iris):
    iris = iris.drop(columns=['species_id','measurement_id'])
    # Encode the species name - not useful as this will be the target
    # dummy_df = pd.get_dummies(iris[['species_name']], dummy_na=False)
    # iris = pd.concat([iris, dummy_df], axis = 1)

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