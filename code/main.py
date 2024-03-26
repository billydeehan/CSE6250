from data_preprocessing import df_characteristics, load_data, clean_data, impute_missing_values, merge_data_sources  
from feature_engineering import encode_features, calculate_feature_importance  
from model_training import split_data, train_model, hyperparameter_tuning, evaluate_model  
from model_evaluation import calculate_metrics, perform_bootstrap_analysis, generate_critical_difference_diagrams  
from model_interpretation import permutation_feature_importance, accumulated_local_effects, local_interpretable_model_explanations  
from deployment_preparation import select_best_model, save_model, create_deployment_pipeline  
import pandas as pd 
def main():  
    # Load and preprocess the dataset  
    df = load_data()
    print(df.head(20)) 
    print(df.describe())
    print(df.info())
    df_characteristics(df)
    df_processed = encode_features(df)
    print(df_processed.head(20)) 
    print(df_processed.describe())
    print(df_processed.info())
    df_characteristics(df_processed)
    
    
    # # Feature engineering  
    # data = encode_categorical_features(data)  
    #feature_importances = calculate_feature_importance(df_processed)  
      
    # # Model training and evaluation  
    # train_set, val_set, test_set = split_data(data)  
    # model, best_params = hyperparameter_tuning(train_set, val_set)  
    # model = train_model(train_set, best_params)  
    # evaluate_model(model, test_set)  
      
    # # Model evaluation and comparison  
    # metrics = calculate_metrics(model, test_set)  
    # perform_bootstrap_analysis(model, test_set)  
    # generate_critical_difference_diagrams(metrics)  
      
    # # Model interpretation  
    # permutation_feature_importance(model, test_set)  
    # accumulated_local_effects(model, test_set)  
    # local_interpretable_model_explanations(model, test_set)  
      
    # # Deployment preparation  
    # best_model = select_best_model(metrics)  
    # save_model(best_model)  
    # create_deployment_pipeline(best_model)  
  
if __name__ == "__main__":  
    main()  