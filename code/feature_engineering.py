import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
from sklearn.ensemble import RandomForestClassifier  
from sklearn.model_selection import train_test_split  
from sklearn.pipeline import Pipeline  
from sklearn.compose import ColumnTransformer  
from sklearn.preprocessing import StandardScaler, OneHotEncoder  
from sklearn.pipeline import Pipeline  
from sklearn.impute import SimpleImputer  

def encode_features(df):  
    
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
    # df['Bleed'] = df['Bleed'].map(yn_mapping)
    # df['Cirrhosis'] = df['Cirrhosis'].map(yn_mapping)
    # df['Surveillance_programme'] = df['Surveillance_programme'].map(yn_mapping)
    # df['Date_incident_surveillance_scan'] = df['Date_incident_surveillance_scan'].map(yn_mapping)
    # df['Prev_known_cirrhosis'] = df['Prev_known_cirrhosis'].map(yn_mapping)

    gender_mapping = {
        'M': 1,
        'F': 0
    }
    df['Gender'] = df['Gender'].map(gender_mapping)
    
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
    df['Year'] = df['Year'].map(year_mapping)
    
    # binary_features = ['Cancer', 'Bleed', 'Year', 'Gender', 'Cirrhosis','Date_incident_surveillance_scan', 'Prev_known_cirrhosis', 'Alive_Dead', 'Surveillance_programme','Prev_known_cirrhosis']
    binary_features = ['Cancer', 'Year', 'Gender', 'Prev_known_cirrhosis', 'Alive_Dead']
    numerical_features = ['Month', 'Age', 'Size', 'Survival_fromMDM','Time_diagnosis_1st_Tx','PS','Time_MDM_1st_treatment','Time_decisiontotreat_1st_treatment','Months_from_last_surveillance']  
    # categorical_features = ['Mode_Presentation', 'Etiology', 'Treatment_grps', 'Type_of_incidental_finding', 'Mode_of_surveillance_detection']  
    categorical_features = ['Bleed', 'Cirrhosis','Prev_known_cirrhosis', 'Surveillance_programme', 'Date_incident_surveillance_scan','Mode_Presentation', 'Etiology', 'Treatment_grps', 'Type_of_incidental_finding', 'Mode_of_surveillance_detection']  
    sequential_features = ['HCC_TNM_Stage', 'HCC_BCLC_Stage', 'ICC_TNM_Stage', 'Surveillance_effectiveness'] 
      
    # Preprocessing for numerical data  
    numerical_transformer = Pipeline(steps=[  
        ('imputer', SimpleImputer(strategy='mean')),  # Handle missing values  
        ('scaler', StandardScaler())  # Standardize numerical features  
    ])  
      
    # Preprocessing for categorical data  
    categorical_transformer = Pipeline(steps=[  
        # ('onehot', OneHotEncoder(handle_unknown='ignore', handle_missing='indicator'))  # One-hot encode categorical features  
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])  
      
    # Preprocessing for binary data  
    binary_transformer = Pipeline(steps=[  
        ('imputer', SimpleImputer(strategy='constant', fill_value=-1)),  # Handle missing values  
        # No need for scaling since these are binary features  
    ])  
    
    # Preprocessing for sequential data  
    sequential_transformer = Pipeline(steps=[  
        ('imputer', SimpleImputer(strategy='mean')),  # Handle missing values  
        # No need for scaling since these are ordinal features  
    ])  
    
    # Add binary and sequential transformers to the preprocessor  
    preprocessor = ColumnTransformer(  
        transformers=[  
            ('num', numerical_transformer, numerical_features),  
            ('cat', categorical_transformer, categorical_features),  
            ('bin', binary_transformer, binary_features),  
            ('seq', sequential_transformer, sequential_features),   
        ], remainder='drop')  
      
    df_processed_array = preprocessor.fit_transform(df)  
    categorical_feature_names = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out() 
    print(categorical_feature_names)
    # Combine all feature names  
    all_feature_names = numerical_features + binary_features + sequential_features + list(categorical_feature_names)  
    print(all_feature_names)
    # Create a DataFrame with the processed data and the correct feature names  
    df_processed = pd.DataFrame(df_processed_array, columns=all_feature_names, index=df.index)  
    
      
    return df_processed  
  
def calculate_feature_importance(df, preprocessor, target_variable='Surveillance_effectiveness'):  
    pass