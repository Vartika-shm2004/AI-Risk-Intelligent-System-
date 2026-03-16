import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

class RiskExplainer:
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None
        
    def create_explainer(self, X_train, model_type='auto'):
        if isinstance(self.model, XGBClassifier) or model_type == 'xgboost':
            self.explainer = shap.TreeExplainer(self.model)
        elif isinstance(self.model, RandomForestClassifier) or model_type == 'rf':
            self.explainer = shap.TreeExplainer(self.model)
        else:
            self.explainer = shap.KernelExplainer(
                lambda x: self.model.predict_proba(x)[:, 1] if hasattr(self.model, 'predict_proba') 
                else self.model.predict(x),
                X_train[:100]
            )
        
        return self.explainer
    
    def calculate_shap_values(self, X):
        if self.explainer is None:
            raise ValueError("Explainer not created. Call create_explainer first.")
        
        if isinstance(self.explainer, shap.KernelExplainer):
            self.shap_values = self.explainer.shap_values(X)
        else:
            self.shap_values = self.explainer.shap_values(X)
        
        return self.shap_values
    
    def get_global_feature_importance(self, X=None):
        if self.shap_values is None:
            if X is not None:
                self.calculate_shap_values(X)
            else:
                raise ValueError("SHAP values not calculated. Provide X to calculate.")
        
        if isinstance(self.shap_values, list):
            shap_values = np.abs(self.shap_values[0])
        else:
            shap_values = np.abs(self.shap_values)
        
        mean_shap = np.mean(shap_values, axis=0)
        
        importance_df = np.zeros(len(self.feature_names))
        for i, name in enumerate(self.feature_names):
            if i < len(mean_shap):
                importance_df[i] = mean_shap[i]
        
        return importance_df
    
    def plot_feature_importance(self, X=None, save_path=None, max_features=15):
        if self.shap_values is None:
            if X is not None:
                self.calculate_shap_values(X)
            else:
                raise ValueError("SHAP values not calculated.")
        
        if isinstance(self.shap_values, list):
            shap_values_plot = self.shap_values[0]
        else:
            shap_values_plot = self.shap_values
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values_plot, 
            features=X[:min(1000, len(X))], 
            feature_names=self.feature_names,
            max_display=max_features,
            show=False
        )
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_beeswarm(self, X=None, save_path=None, max_features=15):
        if self.shap_values is None:
            if X is not None:
                self.calculate_shap_values(X)
            else:
                raise ValueError("SHAP values not calculated.")
        
        if isinstance(self.shap_values, list):
            shap_values_plot = self.shap_values[0]
        else:
            shap_values_plot = self.shap_values
        
        plt.figure(figsize=(12, 10))
        shap.summary_plot(
            shap_values_plot,
            features=X[:min(1000, len(X))],
            feature_names=self.feature_names,
            plot_type="beeswarm",
            max_display=max_features,
            show=False
        )
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def get_local_explanation(self, X_instance, instance_index=0):
        if self.shap_values is None:
            raise ValueError("SHAP values not calculated.")
        
        if isinstance(self.shap_values, list):
            local_shap = self.shap_values[0][instance_index]
        else:
            local_shap = self.shap_values[instance_index]
        
        explanation = []
        for i, (name, value) in enumerate(zip(self.feature_names, local_shap)):
            explanation.append({
                'feature': name,
                'shap_value': value,
                'abs_value': abs(value)
            })
        
        explanation.sort(key=lambda x: x['abs_value'], reverse=True)
        
        return explanation[:10]
    
    def plot_local_explanation(self, X_instance, instance_index=0, save_path=None):
        if self.shap_values is None:
            raise ValueError("SHAP values not calculated.")
        
        if isinstance(self.shap_values, list):
            shap_values_local = self.shap_values[0]
        else:
            shap_values_local = self.shap_values
        
        plt.figure(figsize=(10, 6))
        shap.force_plot(
            self.explainer.expected_value[0] if isinstance(self.explainer.expected_value, list) 
            else self.explainer.expected_value,
            shap_values_local[instance_index],
            X_instance[instance_index],
            feature_names=self.feature_names,
            matplotlib=True,
            show=False
        )
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def create_explanation_dataframe(self, X, top_n=10):
        if self.shap_values is None:
            self.calculate_shap_values(X)
        
        importance = self.get_global_feature_importance(X)
        
        df = pd.DataFrame({
            'Feature': self.feature_names,
            'SHAP Importance': importance
        }).sort_values('SHAP Importance', ascending=False).head(top_n)
        
        return df
