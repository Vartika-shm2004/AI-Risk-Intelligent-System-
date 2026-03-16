import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

class RiskModelTrainer:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.results = {}
        
    def initialize_models(self):
        self.models = {
            'Logistic Regression': LogisticRegression(
                max_iter=1000,
                random_state=42,
                n_jobs=-1
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            ),
            'XGBoost': XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
                eval_metric='logloss'
            )
        }
    
    def train_model(self, model_name, X_train, y_train):
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        model.fit(X_train, y_train)
        return model
    
    def evaluate_model(self, model, X_test, y_test):
        y_pred = model.predict(X_test)
        
        try:
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        except:
            y_pred_proba = y_pred
        
        metrics = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'Recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'F1-Score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        }
        
        try:
            metrics['ROC-AUC'] = roc_auc_score(y_test, y_pred_proba, average='weighted')
        except:
            metrics['ROC-AUC'] = 0.0
            
        metrics['Confusion Matrix'] = confusion_matrix(y_test, y_pred)
        
        return metrics
    
    def train_all_models(self, X, y, test_size=0.2, random_state=42):
        self.initialize_models()
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        results = {}
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            
            model.fit(X_train, y_train)
            
            metrics = self.evaluate_model(model, X_test, y_test)
            results[name] = metrics
            
            try:
                cv_scores = cross_val_score(model, X, y, cv=5, scoring='f1_weighted')
                results[name]['CV F1 Mean'] = cv_scores.mean()
                results[name]['CV F1 Std'] = cv_scores.std()
            except:
                results[name]['CV F1 Mean'] = 0.0
                results[name]['CV F1 Std'] = 0.0
        
        self.results = results
        
        best_score = 0
        for name, metrics in results.items():
            if metrics['F1-Score'] > best_score:
                best_score = metrics['F1-Score']
                self.best_model_name = name
                self.best_model = self.models[name]
        
        return X_train, X_test, y_train, y_test, results
    
    def hyperparameter_tuning(self, model_name, X_train, y_train, param_grid=None):
        if model_name == 'XGBoost':
            if param_grid is None:
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [4, 6, 8],
                    'learning_rate': [0.05, 0.1, 0.2],
                }
            model = XGBClassifier(random_state=42, n_jobs=-1, eval_metric='logloss')
        elif model_name == 'Random Forest':
            if param_grid is None:
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 15],
                    'min_samples_split': [2, 5, 10],
                }
            model = RandomForestClassifier(random_state=42, n_jobs=-1)
        else:
            return None
        
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        return grid_search.best_params_, grid_search.best_score_
    
    def save_model(self, model, filepath):
        joblib.dump(model, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        return joblib.load(filepath)
    
    def predict_risk(self, model, X):
        predictions = model.predict(X)
        
        try:
            probabilities = model.predict_proba(X)
            if probabilities.shape[1] == 2:
                risk_scores = probabilities[:, 1] * 100
            else:
                risk_scores = np.max(probabilities, axis=1) * 100
        except:
            risk_scores = predictions * 100 / 2
        
        return predictions, risk_scores
    
    def classify_risk(self, risk_scores):
        categories = []
        for score in risk_scores:
            if score <= 30:
                categories.append('Low')
            elif score <= 70:
                categories.append('Medium')
            else:
                categories.append('High')
        return categories
