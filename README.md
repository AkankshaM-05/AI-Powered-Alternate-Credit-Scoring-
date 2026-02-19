# AI-Powered-Alternate-Credit-Scoring-
A transparent AI Credit Engine using XGBoost to unlock lending.
This project implements a machine learning credit risk model that evaluates current repayment capacity instead of relying primarily on historical credit records. The objective is to expand access for thin-file and first-time borrowers while preserving institutional risk discipline.

Traditional credit scoring systems depend heavily on prior loan performance, which structurally disadvantages individuals without formal credit history. This framework addresses that limitation by assessing present financial strength using engineered financial indicators and calibrated risk modeling.

The model introduces three core engineered variables — Repayment Burden, Liquidity Ratio, and a log-transformed Risk Multiplier — to capture affordability and financial stress sensitivity. Feature space was reduced from 54 to 19 variables using SHAP-based importance ranking and correlation pruning (>0.75 threshold), improving interpretability and audit readiness. Class imbalance was handled using BorderlineSMOTE, and a strict probability threshold of 0.65 was selected to prioritize precision and capital protection. A controlled Inclusion Boost (+0.10 adjustment) was applied to thin-file applicants, with measured trade-off analysis.

Final model performance:

- Accuracy: 74%
- Precision: 81.8%
- Recall: 80.7%
- ROC–AUC: 0.78
- 5-Fold CV AUC: 0.88
- False Positives: Reduced to 25

The precision-focused threshold reduces capital exposure while maintaining balanced recall and stable generalization.

Governance controls include SHAP explainability, Disparate Impact testing (Four-Fifths Rule), structural sensitivity auditing, calibration curve validation, and cross-validation stability checks. Each approval or rejection can be traced to financially meaningful drivers.

The solution is built using Python 3.10, Pandas, NumPy, Scikit-Learn, XGBoost, Imbalanced-Learn, SHAP, and Matplotlib/Seaborn. It is structured as a modular, deployment-ready pipeline.

The prototype is validated on the German Credit dataset (~1,000 records) with no personally identifiable information used.

This repository presents a disciplined, explainable, and inclusion-aware credit scoring framework suitable for controlled pilot deployment and scalable integration.
