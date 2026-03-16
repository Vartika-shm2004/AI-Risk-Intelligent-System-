# AI Risk Intelligence System

An intelligent, data-driven platform that predicts and evaluates risk using advanced Machine Learning algorithms. This system analyzes structured datasets containing financial, transactional, behavioral, and operational parameters to generate dynamic risk scores and provide actionable insights.

## Features

- **Multiple ML Models**: Logistic Regression, Random Forest, XGBoost
- **Explainable AI**: SHAP values for model interpretability
- **Interactive Dashboard**: Real-time risk visualization with Streamlit
- **Risk Classification**: Low, Medium, and High risk categories
- **Model Evaluation**: Accuracy, Precision, Recall, F1-Score, ROC-AUC

## Project Structure

```
ai-risk-intelligence-system/
├── app.py                      # Main Streamlit dashboard
├── config.py                   # Configuration settings
├── requirements.txt            # Python dependencies
├── generate_data.py            # Sample data generator
├── README.md                   # Project documentation
├── data/
│   └── risk_data.csv          # Sample risk dataset
└── src/
    ├── __init__.py
    ├── data_generator.py      # Data generation module
    ├── preprocessing/
    │   └── data_preprocessor.py
    ├── models/
    │   └── model_trainer.py
    └── utils/
        └── explainer.py       # SHAP explainer
```

## Installation

1. Clone the repository or download the project files.

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Generate Sample Data

The system automatically generates sample data if not present. You can also manually generate:

```bash
python generate_data.py
```

### Run the Dashboard

```bash
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`.

## Dashboard Tabs

1. **Dashboard**: Overview of risk distribution and statistics
2. **Predictions**: Make individual risk predictions with input features
3. **Model Analysis**: Compare model performance metrics
4. **Explainability**: View SHAP feature importance and explanations

## Risk Scoring

- **Low Risk**: 0-30
- **Medium Risk**: 31-70  
- **High Risk**: 71-100

Risk thresholds can be adjusted in the sidebar.

## Technologies Used

- **Python**: Programming language
- **Pandas & NumPy**: Data processing
- **Scikit-learn**: Machine learning models
- **XGBoost**: Gradient boosting
- **SHAP**: Explainable AI
- **Streamlit**: Interactive dashboard
- **Matplotlib/Seaborn**: Visualization

## Sample Data Features

- Transaction amount, frequency
- Credit score, income level
- Debt-to-income ratio
- Payment history score
- Savings/checking balance
- Loan details
- And more...

## Requirements

- Python 3.8+
- 8GB RAM minimum (16GB recommended)
- See requirements.txt for Python packages

## License

This project is for educational purposes.
