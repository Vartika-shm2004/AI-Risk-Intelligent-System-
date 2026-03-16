import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix, classification_report
)

st.set_page_config(
    page_title="AI Risk Intelligence System",
    page_icon="shield",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .stApp {
        background-color: #0e1117;
    }
    .metric-card {
        background-color: #1e293b;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #334155;
    }
    .risk-high {
        color: #ef4444;
        font-weight: bold;
    }
    .risk-medium {
        color: #f59e0b;
        font-weight: bold;
    }
    .risk-low {
        color: #22c55e;
        font-weight: bold;
    }
    h1, h2, h3 {
        color: #e2e8f0;
    }
    .stDataFrame {
        background-color: #1e293b;
    }
</style>
""", unsafe_allow_html=True)

def generate_sample_data(n_samples=10000, random_state=42):
    np.random.seed(random_state)
    
    data = {
        'transaction_id': [f'TXN{i:06d}' for i in range(1, n_samples + 1)],
        'transaction_amount': np.random.exponential(scale=500, size=n_samples),
        'transaction_frequency': np.random.poisson(lam=5, size=n_samples),
        'account_age_days': np.random.randint(30, 3650, size=n_samples),
        'credit_score': np.random.randint(300, 850, size=n_samples),
        'income_level': np.random.randint(20000, 200000, size=n_samples),
        'debt_to_income_ratio': np.random.uniform(0.1, 0.6, size=n_samples),
        'num_prev_defaults': np.random.poisson(lam=0.3, size=n_samples),
        'num_late_payments': np.random.poisson(lam=1, size=n_samples),
        'employment_status': np.random.choice([0, 1, 2], size=n_samples, p=[0.1, 0.8, 0.1]),
        'loan_amount': np.random.exponential(scale=10000, size=n_samples),
        'loan_term_months': np.random.choice([12, 24, 36, 48, 60], size=n_samples),
        'interest_rate': np.random.uniform(3.0, 25.0, size=n_samples),
        'payment_history_score': np.random.uniform(0, 100, size=n_samples),
        'savings_balance': np.random.exponential(scale=5000, size=n_samples),
        'checking_balance': np.random.exponential(scale=2000, size=n_samples),
        'num_accounts': np.random.randint(1, 10, size=n_samples),
        'recent_inquiry_count': np.random.poisson(lam=2, size=n_samples),
        'utilization_rate': np.random.uniform(0, 1, size=n_samples),
        'annual_expenses': np.random.randint(10000, 100000, size=n_samples),
    }
    
    df = pd.DataFrame(data)
    
    risk_score = (
        0.15 * (df['transaction_amount'] / df['transaction_amount'].max()) +
        0.10 * (df['num_prev_defaults'] / df['num_prev_defaults'].max()) +
        0.10 * (df['num_late_payments'] / df['num_late_payments'].max()) +
        0.15 * (1 - df['credit_score'] / 850) +
        0.10 * df['debt_to_income_ratio'] +
        0.10 * (1 - df['payment_history_score'] / 100) +
        0.10 * df['utilization_rate'] +
        0.10 * (df['recent_inquiry_count'] / 10) +
        0.10 * (1 - df['savings_balance'] / df['savings_balance'].max())
    )
    
    risk_score = (risk_score / risk_score.max()) * 100
    
    df['risk_score'] = risk_score
    df['risk_label'] = pd.cut(
        risk_score,
        bins=[-np.inf, 30, 70, np.inf],
        labels=['Low', 'Medium', 'High']
    )
    df['risk_label_encoded'] = df['risk_label'].map({'Low': 0, 'Medium': 1, 'High': 2})
    
    missing_idx = np.random.choice(n_samples, size=int(n_samples * 0.03), replace=False)
    df.loc[missing_idx, 'credit_score'] = np.nan
    
    outlier_idx = np.random.choice(n_samples, size=int(n_samples * 0.02), replace=False)
    df.loc[outlier_idx, 'transaction_amount'] = df['transaction_amount'].max() * 5
    
    return df

def load_data():
    try:
        if not os.path.exists('data/risk_data.csv'):
            os.makedirs('data', exist_ok=True)
            df = generate_sample_data(10000)
            df.to_csv('data/risk_data.csv', index=False)
            st.info("Generated sample data automatically!")
        df = pd.read_csv('data/risk_data.csv')
        return df
    except Exception as e:
        return None

def preprocess_data(df):
    df = df.copy()
    
    exclude_cols = ['transaction_id', 'risk_label', 'risk_label_encoded', 'risk_score']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    
    for col in numeric_cols:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)
    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(df[numeric_cols])
    
    return X, df['risk_label_encoded'], numeric_cols, scaler

def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
        'XGBoost': XGBClassifier(n_estimators=100, max_depth=6, random_state=42, eval_metric='logloss', n_jobs=-1)
    }
    
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        results[name] = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'Recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'F1-Score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        }
        
        try:
            results[name]['ROC-AUC'] = roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovr')
        except:
            results[name]['ROC-AUC'] = 0
            
        trained_models[name] = model
    
    return trained_models, results, X_test, y_test

def plot_confusion_matrix(cm, labels):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels, ax=ax)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    return fig

def plot_metrics_comparison(results):
    models = list(results.keys())
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    
    data = []
    for model in models:
        for metric in metrics:
            data.append({
                'Model': model,
                'Metric': metric,
                'Value': results[model][metric]
            })
    
    df = pd.DataFrame(data)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=df, x='Metric', y='Value', hue='Model', ax=ax)
    plt.title('Model Performance Comparison')
    plt.ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    return fig

def main():
    st.title("🛡️ AI Risk Intelligence System")
    st.markdown("### Intelligent Risk Prediction & Analysis Platform")
    
    df = load_data()
    
    if df is None:
        st.error("Unable to load or generate data.")
        return
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        st.subheader("Model Selection")
        selected_model = st.selectbox(
            "Choose Model",
            ["Random Forest", "XGBoost", "Logistic Regression"]
        )
        
        st.subheader("Risk Thresholds")
        low_threshold = st.slider("Low Risk Max", 0, 50, 30)
        high_threshold = st.slider("High Risk Min", 50, 100, 70)
        
        st.markdown("---")
        st.info("AI-powered risk assessment system using Machine Learning and Explainable AI.")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🔮 Predictions", "📈 Model Analysis", "🔍 Explainability"])
    
    with tab1:
        st.header("Risk Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            high_risk = (df['risk_label'] == 'High').sum()
            st.metric("High Risk Cases", high_risk, f"{high_risk/len(df)*100:.1f}%")
        with col3:
            medium_risk = (df['risk_label'] == 'Medium').sum()
            st.metric("Medium Risk Cases", medium_risk, f"{medium_risk/len(df)*100:.1f}%")
        with col4:
            low_risk = (df['risk_label'] == 'Low').sum()
            st.metric("Low Risk Cases", low_risk, f"{low_risk/len(df)*100:.1f}%")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Risk Distribution")
            fig, ax = plt.subplots(figsize=(8, 5))
            colors = {'Low': '#22c55e', 'Medium': '#f59e0b', 'High': '#ef4444'}
            df['risk_label'].value_counts().plot(kind='bar', color=[colors.get(x, '#6b7280') for x in df['risk_label'].value_counts().index], ax=ax)
            plt.title("Risk Category Distribution")
            plt.xlabel("Risk Category")
            plt.ylabel("Count")
            plt.xticks(rotation=0)
            st.pyplot(fig)
        
        with col2:
            st.subheader("Risk Score Distribution")
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(df['risk_score'], bins=50, color='#3b82f6', edgecolor='white', alpha=0.7)
            ax.axvline(x=low_threshold, color='#22c55e', linestyle='--', label=f'Low Risk: 0-{low_threshold}')
            ax.axvline(x=high_threshold, color='#ef4444', linestyle='--', label=f'High Risk: {high_threshold}-100')
            plt.xlabel("Risk Score")
            plt.ylabel("Frequency")
            plt.legend()
            st.pyplot(fig)
        
        st.subheader("Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
    
    with tab2:
        st.header("Risk Prediction")
        
        with st.spinner("Training models..."):
            X, y, feature_names, scaler = preprocess_data(df)
            trained_models, results, X_test, y_test = train_models(X, y)
        
        st.success("Models trained successfully!")
        
        st.subheader("Make a Prediction")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            trans_amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=500.0)
            trans_freq = st.number_input("Transaction Frequency", min_value=0, value=5)
            account_age = st.number_input("Account Age (days)", min_value=0, value=365)
            credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
            income = st.number_input("Annual Income ($)", min_value=0, value=50000)
            debt_ratio = st.slider("Debt-to-Income Ratio", 0.0, 1.0, 0.3)
        
        with col2:
            defaults = st.number_input("Previous Defaults", min_value=0, value=0)
            late_payments = st.number_input("Late Payments", min_value=0, value=0)
            employment_status = st.selectbox("Employment Status", [0, 1, 2], format_func=lambda x: ["Employed", "Self-Employed", "Unemployed"][x])
            loan_amount = st.number_input("Loan Amount ($)", min_value=0.0, value=10000.0)
            loan_term = st.selectbox("Loan Term (months)", [12, 24, 36, 48, 60])
            interest_rate = st.slider("Interest Rate (%)", 3.0, 25.0, 12.0)
        
        with col3:
            payment_score = st.slider("Payment History Score", 0, 100, 75)
            savings_balance = st.number_input("Savings Balance ($)", min_value=0.0, value=5000.0)
            checking_balance = st.number_input("Checking Balance ($)", min_value=0.0, value=2000.0)
            num_accounts = st.number_input("Number of Accounts", min_value=1, max_value=20, value=3)
            inquiry_count = st.number_input("Recent Inquiry Count", min_value=0, value=2)
            utilization = st.slider("Credit Utilization", 0.0, 1.0, 0.5)
        
        if st.button("Predict Risk", type="primary"):
            model = trained_models[selected_model]
            
            input_features = [
                trans_amount, trans_freq, account_age, credit_score, income,
                debt_ratio, defaults, late_payments, employment_status,
                loan_amount, loan_term, interest_rate, payment_score,
                savings_balance, checking_balance, num_accounts, inquiry_count,
                utilization, income * 0.5
            ]
            
            input_data = np.array([input_features])
            input_scaled = scaler.transform(input_data)
            
            prediction = model.predict(input_scaled)[0]
            proba = model.predict_proba(input_scaled)[0]
            
            risk_score = proba[prediction] * 100
            
            if risk_score <= low_threshold:
                risk_level = "Low"
            elif risk_score <= high_threshold:
                risk_level = "Medium"
            else:
                risk_level = "High"
            
            st.markdown("---")
            st.subheader("Prediction Result")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Risk Level", risk_level)
            with col2:
                st.metric("Risk Score", f"{risk_score:.1f}")
            with col3:
                confidence = max(proba) * 100
                st.metric("Confidence", f"{confidence:.1f}%")
            
            st.progress(risk_score / 100)
            
            st.subheader("Risk Probability Distribution")
            proba_df = pd.DataFrame({
                'Risk Level': ['Low', 'Medium', 'High'],
                'Probability': proba * 100
            })
            st.bar_chart(proba_df.set_index('Risk Level'))
    
    with tab3:
        st.header("Model Performance Analysis")
        
        with st.spinner("Training models..."):
            X, y, feature_names, scaler = preprocess_data(df)
            trained_models, results, X_test, y_test = train_models(X, y)
        
        st.subheader("Performance Metrics Comparison")
        st.pyplot(plot_metrics_comparison(results))
        
        st.subheader("Model Metrics")
        metrics_df = pd.DataFrame(results).T
        st.dataframe(metrics_df.style.background_gradient(cmap='Blues'), use_container_width=True)
        
        st.subheader("Confusion Matrix")
        model = trained_models[selected_model]
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        st.pyplot(plot_confusion_matrix(cm, ['Low', 'Medium', 'High']))
        
        st.subheader("Classification Report")
        report = classification_report(y_test, y_pred, target_names=['Low', 'Medium', 'High'])
        st.text(report)
    
    with tab4:
        st.header("Explainable AI (SHAP)")
        st.markdown("Understanding model predictions with SHAP values")
        
        try:
            with st.spinner("Computing SHAP values..."):
                X, y, feature_names, scaler = preprocess_data(df)
                trained_models, results, X_test, y_test = train_models(X, y)
                model = trained_models[selected_model]
                
                if isinstance(model, (RandomForestClassifier, XGBClassifier)):
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X_test[:300])
                    
                    st.subheader("Feature Importance (SHAP)")
                    fig, ax = plt.subplots(figsize=(10, 8))
                    if isinstance(shap_values, list):
                        shap.summary_plot(shap_values[0], X_test[:300], 
                                         feature_names=feature_names, show=False)
                    else:
                        shap.summary_plot(shap_values, X_test[:300], 
                                         feature_names=feature_names, show=False)
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    st.subheader("Top 10 Important Features")
                    if isinstance(shap_values, list):
                        importance = np.abs(shap_values[0]).mean(axis=0)
                    else:
                        importance = np.abs(shap_values).mean(axis=0)
                    
                    if importance.ndim > 1:
                        importance = importance.flatten()
                    
                    n_features = min(len(feature_names), len(importance))
                    importance_df = pd.DataFrame({
                        'Feature': list(feature_names[:n_features]),
                        'Importance': list(importance[:n_features])
                    }).sort_values('Importance', ascending=False).head(10)
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.barplot(data=importance_df, y='Feature', x='Importance', ax=ax, palette='viridis')
                    plt.title("Top 10 Feature Importance")
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.warning(f"SHAP explanations are best supported for tree-based models. Current: {selected_model}")
                    
                    importance_df = pd.DataFrame({
                        'Feature': list(feature_names),
                        'Importance': list(np.abs(model.coef_[0]) if hasattr(model, 'coef_') else np.zeros(len(feature_names)))
                    }).sort_values('Importance', ascending=False).head(10)
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.barplot(data=importance_df, y='Feature', x='Importance', ax=ax, palette='viridis')
                    plt.title("Feature Importance (Linear Coefficients)")
                    plt.tight_layout()
                    st.pyplot(fig)
        except Exception as e:
            st.error(f"Error computing SHAP values: {str(e)}")
            st.info("Try selecting a different model (XGBoost or Random Forest recommended)")

if __name__ == "__main__":
    main()
