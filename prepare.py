import pandas as pd
from env import get_db_url


def prep_iris(iris):
    iris = iris.drop(columns=['species_id','measurement_id'])
    dummy_df = pd.get_dummies(iris[['species_name']], dummy_na=False)
    iris = pd.concat([iris, dummy_df], axis = 1)

    return iris

def prep_titantic(df):
    df = df.drop(columns=['passenger_id','embarked','deck','class','age'])
    dummy_df = pd.get_dummies(df[['sex','embark_town']], dummy_na=False, drop_first=True)
    df = pd.concat([df,dummy_df], axis = 1)
    
    return df.drop(columns=['sex','embark_town'])

def prep_telco(df):
    
    df = df.dropna()
    df = df.drop(columns=['payment_type_id','internet_service_type_id','contract_type_id'])
    catcol = df.columns[(df.nunique()<5)&(df.dtypes == 'object')]
    dummy_df = pd.get_dummies(df[catcol], dummy_na=False, drop_first=True)
    df = pd.concat([df,dummy_df],axis=1)
    df = df.drop(columns=catcol)
    
    return df