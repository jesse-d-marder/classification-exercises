import pandas as pd
from env import get_db_url
import os

def get_titantic_data():
    """Return titanic data from codeup data science database as a pandas data frame"""
    filename = 'titantic.csv'
    if os.path.exists(filename):
        print("Using cached data")
        return pd.read_csv(filename)

    query = '''
        SELECT *
        FROM passengers'''

    df = pd.read_sql(query,get_db_url('titanic_db'))
    df.to_csv(filename, index=False)
    return df

def get_iris_data():
    """Return iris data from codeup data science database as a pandas data frame"""
    filename = 'iris.csv'
    if os.path.exists(filename):
        print("Using cached data")
        return pd.read_csv(filename)
    
    query = '''
        SELECT *
        FROM measurements
        JOIN species
        USING (species_id)'''

    df = pd.read_sql(query,get_db_url('iris_db'))
    df.to_csv(filename, index=False)
    return df

def get_telco_data():
    """Return data from telco_churn database in SQL as a pandas data frame"""
    filename = 'telco.csv'
    if os.path.exists(filename):
        print("Using cached data")
        return pd.read_csv(filename)
    
    query = '''
        SELECT *
        FROM customers
        JOIN contract_types
        USING (contract_type_id)
        JOIN internet_service_types
        USING (internet_service_type_id)
        JOIN payment_types
        USING (payment_type_id)'''
        
    df = pd.read_sql(query,get_db_url('telco_churn'))
    df.to_csv(filename, index=False)
    return df

