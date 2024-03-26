import pandas as pd


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


  