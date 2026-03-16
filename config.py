RISK_THRESHOLDS = {
    'low': 30,
    'medium': 70,
    'high': 100
}

MODEL_CONFIG = {
    'logistic_regression': {
        'max_iter': 1000,
        'random_state': 42
    },
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5,
        'random_state': 42
    },
    'xgboost': {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'random_state': 42,
        'eval_metric': 'logloss'
    }
}

PREPROCESSING_CONFIG = {
    'missing_value_strategy': 'median',
    'outlier_method': 'iqr',
    'outlier_threshold': 1.5,
    'scaling_method': 'standard',
    'test_size': 0.2,
    'random_state': 42
}

DATA_CONFIG = {
    'default_path': 'data/risk_data.csv',
    'encoding': 'utf-8',
    'numeric_precision': 2
}

DASHBOARD_CONFIG = {
    'page_title': 'AI Risk Intelligence System',
    'page_icon': 'shield',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded'
}
