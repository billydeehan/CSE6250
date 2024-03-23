import pandas as pd
import os

from sklearn.preprocessing import StandardScaler, OneHotEncoder  
from sklearn.compose import ColumnTransformer  
from sklearn.pipeline import Pipeline  
from sklearn.impute import SimpleImputer  

def load_data():  
    print('test')
    file_path = 'data/covid-liver.csv'
    df = pd.read_csv(file_path)
    # print(df.head())
    return df

def clean_data(): 
    pass 
def impute_missing_values():
    pass
def merge_data_sources():
    pass
import pandas as pd  
  
def df_characteristics(df):  
    if not isinstance(df, pd.DataFrame):  
        raise TypeError("Input must be a pandas DataFrame.")  
      
    # Iterate through each column in the DataFrame  
    for column in df.columns:  
        print(f"Feature: {column}")  
        print(f"Data Type: {df[column].dtype}")  
          
        # If the feature is numerical, print range and average  
        if pd.api.types.is_numeric_dtype(df[column]):  
            print(f"Range: {df[column].min()} to {df[column].max()}")  
            print(f"Average: {df[column].mean()}")  
          
        # If the feature is categorical, print unique categories  
        elif pd.api.types.is_categorical_dtype(df[column]) or pd.api.types.is_object_dtype(df[column]):  
            print(f"Unique Categories: {df[column].nunique()}")  
            print(f"Categories: {df[column].unique()}")  
          
        # Handle datetime data types  
        elif pd.api.types.is_datetime64_any_dtype(df[column]):  
            print(f"Date Range: {df[column].min()} to {df[column].max()}")  
          
        # Add other data types handling as needed  
          
        print("\n")  
  
# Example usage:  
# Assuming you have a pandas DataFrame named `your_dataframe`, you would call the function like this:  
# print_dataframe_characteristics(your_dataframe)  

  
def preprocess_data(df):  
    
    stage_mapping = {  
        'I': 1,  
        'II': 2,          
        'IIIA+IIIB': 3,
        'III': 3,
        'IV': 4
    }  
    df['HCC_TNM_Stage'] = df['HCC_TNM_Stage'].map(stage_mapping)  
    df['ICC_TNM_Stage'] = df['ICC_TNM_Stage'].map(stage_mapping)  
    
    abc_mapping =  {
        "0": 0,
        "A": 1,
        "B": 2,
        "C": 3,
        "D": 4
    }
    df['HCC_BCLC_Stage'] = df['HCC_BCLC_Stage'].map(abc_mapping)
    
    yn_mapping = {
        'Y': 1,
        'N': 0
    }
    df['Cancer'] = df['Cancer'].map(yn_mapping)
    df['Bleed'] = df['Bleed'].map(yn_mapping)
    df['Cirrhosis'] = df['Cirrhosis'].map(yn_mapping)
    df['Surveillance_programme'] = df['Surveillance_programme'].map(yn_mapping)
    df['Date_incident_surveillance_scan'] = df['Date_incident_surveillance_scan'].map(yn_mapping)
    df['Prev_known_cirrhosis'] = df['Prev_known_cirrhosis'].map(yn_mapping)

    gender_mapping = {
        'M': 1,
        'F': 0
    }
    df['gender'] = df['gender'].map(gender_mapping)
    
    alive_dead_mapping = {
        'Alive': 1,
        'Dead': 0
    }
    df['Alive_Dead'] = df['Alive_Dead'].map(alive_dead_mapping)
    
    effective_mapping = {
        'Missed': 0,
        'Inconsistent': 1,
        'Consistent': 2
    }
    df['Surveillance_effectiveness'] = df['Surveillance_effectiveness'].map(effective_mapping)
    
    year_mapping = {
        'Pandemic': 1,
        'Prepandemic': 0
    }
    
    # List of numerical, categorical, and sequential feature names  
    binary_features = ['Cancer', 'Bleed', 'Cirrhosis', 'Surveillance_programme','Date_incident_surveillance_scan', 'Prev_known_cirrhosis', 'gender', 'Alive_Dead']
    numerical_features = ['Month', 'numerical_column2']  
    
    categorical_features = ['Year', 'categorical_column2']  
    sequential_features = ['sequential_column']  # This can be merged with categorical or numerical as needed  
      
    # Preprocessing for numerical data  
    numerical_transformer = Pipeline(steps=[  
        ('imputer', SimpleImputer(strategy='mean')),  # Handle missing values  
        ('scaler', StandardScaler())  # Standardize numerical features  
    ])  
      
    # Preprocessing for categorical data  
    categorical_transformer = Pipeline(steps=[  
        ('imputer', SimpleImputer(strategy='most_frequent')),  # Handle missing values  
        ('onehot', OneHotEncoder(handle_unknown='ignore'))  # One-hot encode categorical features  
    ])  
      
    # Combine preprocessing for numerical and categorical data  
    preprocessor = ColumnTransformer(  
        transformers=[  
            ('num', numerical_transformer, numerical_features),  
            ('cat', categorical_transformer, categorical_features),  
            # Add more transformers as needed  
        ]  
    )  
      
    # Apply the preprocessing  
    df_processed = preprocessor.fit_transform(df)  
      
    # The output is a NumPy array; convert it back to a DataFrame if needed  
    # Feature names after one-hot encoding will not be preserved  
    df_processed = pd.DataFrame(df_processed)  
      
    return df_processed  
  
# Example usage:  
# df = pd.read_csv('your_data.csv')  
# df_ready_for_ml = preprocess_dataframe(df)  
