import pandas as pd
import os

def load_data():  
    print('test')
    file_path = 'data/covid-liver.csv'
    df = pd.read_csv(file_path, sep='\t')
    print(df.head())
    # return df
    return 'test'
def clean_data(): 
    pass 
def impute_missing_values():
    pass
def merge_data_sources():
    pass

df = load_data()

print(os.getcwd())