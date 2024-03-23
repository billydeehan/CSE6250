https://www.kaggle.com/datasets/fedesoriano/covid19-effect-on-liver-cancer-prediction-dataset

redicting Treatment Adherence of Tuberculosis Patients at Scale
│  
├── data_preprocessing.py  
│   ├── load_data()  
│   ├── clean_data()  
│   ├── impute_missing_values()  
│   └── merge_data_sources()  
│  
├── feature_engineering.py  
│   ├── encode_categorical_features()  
│   └── calculate_feature_importance()  
│  
├── model_training.py  
│   ├── split_data()  
│   ├── train_model()  
│   ├── hyperparameter_tuning()  
│   └── evaluate_model() 
│  
├── model_evaluation.py  
│   ├── calculate_metrics()  
│   ├── perform_bootstrap_analysis()  
│   └── generate_critical_difference_diagrams()  
│  
├── model_interpretation.py  
│   ├── permutation_feature_importance()  
│   ├── accumulated_local_effects()  
│   └── local_interpretable_model_explanations()  
│  
├── deployment_preparation.py  
│   ├── select_best_model()  
│   ├── save_model()  
│   └── create_deployment_pipeline()  
└── main.py  
        └── main()  
