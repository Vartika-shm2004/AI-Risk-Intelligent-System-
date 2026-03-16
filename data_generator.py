import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

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

def main():
    df = generate_sample_data(10000)
    df.to_csv('data/risk_data.csv', index=False)
    print("Sample dataset generated: data/risk_data.csv")
    print(f"Shape: {df.shape}")
    print(f"\nRisk distribution:\n{df['risk_label'].value_counts()}")

if __name__ == "__main__":
    main()
