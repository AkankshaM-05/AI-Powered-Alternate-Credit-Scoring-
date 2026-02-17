# AI-Powered-Alternate-Credit-Scoring-
A transparent AI Credit Engine using XGBoost to unlock lending.
This project builds a machine learning–based alternate credit scoring prototype to evaluate borrowers with limited or no traditional credit history. The goal is to move from purely history-based scoring to a more forward-looking, capacity-based risk assessment model.

In the notebook, we implemented a complete ML pipeline using Python (3.10+) and publicly available, anonymized datasets such as German Credit and Give Me Some Credit. We performed data cleaning, preprocessing, and feature engineering using pandas and numpy. We engineered behavioral financial ratios such as Repayment Burden, Liquidity Ratio, and a Risk Multiplier to better capture real-time repayment capacity.

To address class imbalance, we applied BorderlineSMOTE. We trained multiple models including Logistic Regression (baseline), Random Forest, and XGBoost as the primary model. Model evaluation was done using Stratified 5-Fold Cross-Validation, and the final model achieved a mean AUC-ROC of 0.8521. We optimized the decision threshold to 0.65 to reduce false negatives and prevent additional potential defaults.

For transparency, we implemented SHAP to generate global feature importance and per-prediction explanations. We also conducted a basic bias audit using the Disparate Impact Ratio to check for demographic disparities. The result is a transparent, auditable, and explainable credit scoring prototype designed with financial inclusion in mind.
