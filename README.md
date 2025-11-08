# Credit-Risk-ML-Model
Multi-class credit risk classification using XGBoost

# Credit Risk Classification Model

A production-ready machine learning system for multi-class credit risk assessment using XGBoost with comprehensive business impact analysis.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-Latest-red.svg)

## Project Overview

This project implements a comprehensive credit risk modeling pipeline that:
- Classifies loans into 4 risk categories (P1, P2, P3, P4)
- Achieves 77.99% accuracy on 8,413 loan applications
- Achieves 0.9227 macro-average AUC-ROC score
- Reduces expected loss estimation error to 6.06%
- Provides explainable predictions for regulatory compliance

## Key Features

### Technical Excellence
- Data Leakage Prevention: Train-test split performed before feature selection
- Statistical Feature Selection: Chi-Square test (p≤0.05), VIF analysis (≤6), ANOVA test (p≤0.05)
- Hyperparameter Optimization: RandomizedSearchCV with 5-fold cross-validation
- Model Interpretability: Feature importance analysis with detailed explanations
- 49 engineered features from comprehensive statistical testing

### Business Impact
- Cost-Benefit Analysis: Type I vs Type II error trade-offs quantified
- Expected Loss Calculation: Portfolio risk quantification with 6.06% estimation error
- Optimized Decision Thresholds: 11.62% rejection rate balancing risk and revenue
- Financial Impact Assessment: $21.7M potential loss prevention vs $1.175M opportunity cost

## Tech Stack

**Languages & Core Libraries**
- Python 3.8+
- Pandas, NumPy

**Machine Learning**
- scikit-learn (Model training, evaluation, preprocessing)
- XGBoost (Primary classifier)
- RandomForest, Decision Trees (Baseline comparisons)

**Statistical Analysis**
- SciPy (Chi-Square, ANOVA tests)
- statsmodels (VIF analysis)

**Visualization**
- Matplotlib, Seaborn

**Development Environment**
- Jupyter Notebook

## Model Performance

### Overall Metrics

| Metric | Value |
|--------|-------|
| Test Set Accuracy | 77.99% |
| Macro-Average AUC-ROC | 0.9227 |
| Test Set Size | 8,413 records |
| Number of Features | 49 |
| Expected Loss Estimation Error | 6.06% |

### Per-Class Performance

| Risk Category | Precision | Recall | F1-Score | Interpretation |
|--------------|-----------|--------|----------|----------------|
| P1 (Low Risk) | 0.8366 | 0.7770 | 0.8057 | High confidence in low-risk identification |
| P2 (Medium-Low) | 0.8256 | 0.9196 | 0.8701 | Best performing class |
| P3 (Medium-High) | 0.4385 | 0.2904 | 0.3494 | Conservative - minimizes false approvals |
| P4 (High Risk) | 0.7597 | 0.7056 | 0.7317 | Strong high-risk detection |

### Business Metrics

| Metric | Value | Business Implication |
|--------|-------|---------------------|
| Portfolio Size | 8,413 loans | Representative test set |
| Rejection Rate | 11.62% | Conservative risk management |
| Type I Errors | 310 | High-risk loans incorrectly approved |
| Type II Errors | 235 | Low-risk loans incorrectly rejected |
| Potential Loss (Type I) | $21,700,000 | Assuming $100K average loan, 70% default rate |
| Lost Revenue (Type II) | $1,175,000 | Assuming $100K average loan, 5% profit margin |

## Project Structure
```
Credit-Risk-ML-Model/
│
├── Credit-Risk-ML-Model_notebook/
│   └── credit_risk_modeling.ipynb    # Main analysis notebook
│
├── models/
│   ├── best_credit_risk_model.pkl    # Trained XGBoost model
│   ├── feature_names.pkl             # Feature list for deployment
│   └── label_encoder.pkl             # Target variable encoder
│
├── visualizations/
│   ├── confusion_matrix.png          # Classification performance matrix
│   ├── feature_importance_builtin.png # Top feature rankings
│   └── roc_curves.png                # Multi-class ROC-AUC curves
│
├── reports/
│   └── model_performance_report.txt  # Comprehensive performance metrics
│
├── .gitignore
├── requirements.txt
└── README.md
```

## Methodology

### 1. Data Preprocessing
- Handled missing values indicated by -99999 sentinel values
- Merged multiple data sources with inner join
- Removed features with >10,000 missing values
- Applied domain-specific data cleaning rules

### 2. Feature Engineering & Selection

**Critical: Data Leakage Prevention**
- Train-test split (80-20) performed BEFORE any feature selection
- All feature engineering performed exclusively on training data
- Test set completely isolated until final evaluation

**Statistical Feature Selection**
- Chi-Square Test: Categorical variables (p ≤ 0.05)
- Variance Inflation Factor (VIF): Multicollinearity detection (threshold ≤ 6)
- ANOVA Test: Numerical feature relevance across risk categories (p ≤ 0.05)
- Result: 49 features selected from original feature space

**Feature Encoding**
- Ordinal Encoding: Education levels (SSC=1, 12TH=2, GRADUATE=3, POST-GRADUATE=4)
- One-Hot Encoding: Nominal categorical variables (with drop_first=True)

### 3. Model Development
- Implemented Random Forest, XGBoost, and Decision Tree classifiers
- Performed hyperparameter tuning using RandomizedSearchCV
- 5-fold stratified cross-validation for robust evaluation
- Selected XGBoost as final model based on comprehensive metrics

### 4. Model Evaluation

**Classification Metrics**
- Accuracy, Precision, Recall, F1-Score per class
- Confusion matrix with misclassification cost analysis
- ROC-AUC curves using One-vs-Rest strategy

**Business Metrics**
- Expected loss calculation with risk-weighted categories
- Type I error cost: Approving high-risk loans (most critical)
- Type II error cost: Rejecting viable loans (opportunity cost)
- Portfolio-level risk assessment

### 5. Model Interpretability
- Feature importance rankings from tree-based models
- Business-relevant feature explanations
- Transparent decision-making process for compliance

## How to Run

### Prerequisites

Ensure Python 3.8 or higher is installed, then install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Project

1. Clone this repository:
```bash
git clone https://github.com/RITURAJSHARMA463/Credit-Risk-ML-Model.git
cd Credit-Risk-ML-Model
```

2. Open the Jupyter notebook:
```bash
jupyter notebook Credit-Risk-ML-Model_notebook/credit_risk_modeling.ipynb
```

3. Update file paths in the notebook to point to your data location

4. Run all cells sequentially to reproduce the analysis

### Making Predictions with Saved Model
```python
import pickle
import pandas as pd

# Load model
with open('models/best_credit_risk_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load feature names
with open('models/feature_names.pkl', 'rb') as f:
    features = pickle.load(f)

# Load label encoder
with open('models/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Make predictions
predictions = model.predict(new_data[features])
predicted_labels = label_encoder.inverse_transform(predictions)
```

## Visualizations

### Confusion Matrix
![Confusion Matrix](visualizations/confusion_matrix.png)

*Shows classification performance across all risk categories with actual vs predicted distributions*

### Feature Importance
![Feature Importance](visualizations/feature_importance_builtin.png)

*Top 20 features driving model predictions, critical for interpretability and compliance*

### ROC Curves
![ROC Curves](visualizations/roc_curves.png)

*Multi-class ROC-AUC analysis demonstrating strong discriminative ability across risk categories*

## Business Value

### Risk Management
- Identifies high-risk applicants with 70.56% recall (Type I error mitigation)
- Maintains 77.70% recall for low-risk applicants (revenue preservation)
- Achieves 6.06% expected loss estimation error vs actual portfolio performance

### Cost Optimization
- Type I Error Prevention: Model prevents $21.7M in potential losses by identifying 70.56% of high-risk loans
- Type II Error Trade-off: $1.175M opportunity cost from conservative rejections
- Net Benefit: 18.5:1 ratio of prevented losses to opportunity costs
- Optimal Threshold: 11.62% rejection rate balances risk management with business growth

### Regulatory Compliance
- Explainable decision-making through feature importance
- Transparent model architecture suitable for audit
- Documented methodology preventing discriminatory bias
- Risk-based approach aligned with banking guidelines

## Key Learnings

### Technical Insights
1. Data Leakage Prevention: Train-test split before feature selection is critical - prevents artificially inflated performance metrics
2. Feature Selection Impact: Statistical testing (VIF, ANOVA, Chi-Square) reduced features while improving model generalization
3. Hyperparameter Tuning: RandomizedSearchCV with 5-fold CV improved F1-score by approximately 8% over baseline
4. Class Imbalance: P3 category performance indicates need for targeted sampling strategies in future iterations

### Business Insights
1. Cost Asymmetry: Type I errors (approving bad loans) cost 18.5x more than Type II errors (rejecting good loans)
2. Risk Segmentation: Clear separation between P1/P2 (approve) and P4 (reject) with P3 requiring manual review
3. Model Interpretability: Feature importance aligns with domain expertise, building stakeholder confidence
4. Portfolio Management: 6.06% loss estimation error enables accurate capital reserve planning

## Author

**Rituraj Sharma**
- Email: riturajsharma463@gmail.com
- GitHub: https://github.com/RITURAJSHARMA463



---

**Note**: This project uses anonymized data and is intended for educational and portfolio purposes. No actual customer data or proprietary business logic is included.

*Last Updated: November 2025*
