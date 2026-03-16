import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer

class DataPreprocessor:
    def __init__(self, scaling_method='standard'):
        self.scaling_method = scaling_method
        self.scaler = None
        self.imputer = None
        self.feature_names = None
        
    def handle_missing_values(self, df, strategy='median'):
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        self.imputer = SimpleImputer(strategy=strategy)
        df[numeric_cols] = self.imputer.fit_transform(df[numeric_cols])
        return df
    
    def handle_outliers(self, df, method='iqr', threshold=1.5):
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                df[col] = df[col].clip(lower_bound, upper_bound)
            elif method == 'zscore':
                mean = df[col].mean()
                std = df[col].std()
                df[col] = df[col].clip(mean - threshold * std, mean + threshold * std)
        
        return df
    
    def remove_duplicates(self, df):
        return df.drop_duplicates()
    
    def encode_categorical(self, df, columns=None):
        if columns is None:
            columns = df.select_dtypes(include=['object', 'category']).columns
        
        encoders = {}
        for col in columns:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                encoders[col] = le
        
        return df, encoders
    
    def scale_features(self, X):
        if self.scaling_method == 'standard':
            self.scaler = StandardScaler()
        elif self.scaling_method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            return X
        
        X_scaled = self.scaler.fit_transform(X)
        return X_scaled
    
    def select_features(self, X, feature_names=None):
        if feature_names is not None:
            self.feature_names = feature_names
        return X
    
    def create_derived_features(self, df):
        df = df.copy()
        
        if 'transaction_amount' in df.columns and 'income_level' in df.columns:
            df['amount_to_income_ratio'] = df['transaction_amount'] / (df['income_level'] + 1)
        
        if 'savings_balance' in df.columns and 'income_level' in df.columns:
            df['savings_to_income_ratio'] = df['savings_balance'] / (df['income_level'] + 1)
        
        if 'loan_amount' in df.columns and 'income_level' in df.columns:
            df['loan_to_income_ratio'] = df['loan_amount'] / (df['income_level'] + 1)
        
        if 'num_prev_defaults' in df.columns and 'num_late_payments' in df.columns:
            df['total_negative_history'] = df['num_prev_defaults'] + df['num_late_payments']
        
        if 'checking_balance' in df.columns and 'savings_balance' in df.columns:
            df['total_liquidity'] = df['checking_balance'] + df['savings_balance']
        
        return df
    
    def preprocess(self, df, target_column=None, remove_outliers=True):
        df = df.copy()
        
        df = self.remove_duplicates(df)
        
        if df.select_dtypes(include=[np.number]).isnull().any().any():
            df = self.handle_missing_values(df, strategy='median')
        
        if remove_outliers:
            df = self.handle_outliers(df, method='iqr')
        
        df = self.create_derived_features(df)
        
        exclude_cols = []
        if target_column:
            exclude_cols.append(target_column)
        if 'transaction_id' in df.columns:
            exclude_cols.append('transaction_id')
        if 'risk_label' in df.columns:
            exclude_cols.append('risk_label')
            
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        X = df[feature_cols].select_dtypes(include=[np.number])
        
        self.feature_names = X.columns.tolist()
        X_scaled = self.scale_features(X)
        
        y = None
        if target_column and target_column in df.columns:
            y = df[target_column]
        
        return X_scaled, y, X
    
    def get_feature_names(self):
        return self.feature_names
