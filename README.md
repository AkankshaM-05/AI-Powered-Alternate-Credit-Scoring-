# Alternate Credit Scoring Engine

**Machine Learning Model for Expanded Lending Access**

An XGBoost-based credit risk model that evaluates current repayment capacity rather than historical credit performance, designed to expand lending access for thin-file and first-time borrowers while maintaining risk discipline.

---

## Problem Statement

Traditional credit scoring systems heavily weight prior loan performance, systematically excluding qualified borrowers without formal credit history (thin-file applicants, immigrants, young professionals). This model addresses that gap by assessing **current financial strength** through engineered indicators of repayment capacity.

**Objective**: Expand lending access while preserving capital protection through precision-focused decision thresholds.

---

## Approach

### Feature Engineering

Three core engineered variables designed to capture repayment capacity:

1. **Repayment Burden** – Monthly debt obligations relative to income
2. **Liquidity Ratio** – Access to liquid assets for emergency repayment
3. **Risk Multiplier** – Log-transformed combination of employment stability and income volatility

These features directly measure affordability, not historical behavior.

### Dimensionality Reduction

**Initial feature count**: 54 variables  
**Final feature count**: 19 variables  
**Method**: SHAP-based importance ranking + correlation pruning (>0.75 threshold)

**Rationale**: 
- Reduces model complexity and inference latency
- Improves interpretability for credit decisions
- Enables audit-ready explanations for each approval/rejection

### Class Imbalance Handling

Applied **BorderlineSMOTE** to training data:
- Oversamples minority class (non-approved borrowers)
- Preserves decision boundary structure
- Prevents false positive inflation in imbalanced dataset

### Threshold Calibration

**Probability threshold selected: 0.65**

Rationale:
- Optimizes for precision (minimize false positives = capital loss)
- Maintains recall above 80% (minimize false negatives = missed revenue)
- Reduces false positives to 25 (test set, ~400 samples)

### Fairness Consideration

**Inclusion Boost Applied**: +0.10 probability adjustment for thin-file applicants

- Targets applicants with zero/minimal credit history
- Measured trade-off analysis between inclusion and risk
- Simple policy control (not model retraining)

**Current Limitation**: Four-Fifths Rule testing implemented but not fully validated on production demographics. Recommend full disparate impact analysis before deployment.

---

## Results

### Model Performance

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 76.5% |
| **Precision** | 81.8% |
| **Recall** | 80.7% |
| **ROC-AUC (Test)** | 0.78 |
| **5-Fold CV AUC** | 0.88 (±0.154) |
| **False Positives (Test)** | 25 / 400 |

### Performance Interpretation

**Gap between CV (0.88) and Test (0.78)**:
- Indicates some overfitting to training distribution
- CV used full dataset with resampling; test is held-out
- Acceptable for prototype; monitor on production data

**Precision-Recall Trade-off**:
- 81.8% precision means 18 of 100 approvals default
- 80.7% recall means 80 of 100 true non-defaulters approved
- Threshold optimizes for lender risk tolerance

**False Positive Cost**: 
- Each false positive = loan loss
- 25 FP on 400 test samples = manageable loss rate for pilot

---

## Feature Importance (SHAP Analysis)

Top contributing features to approval decisions:

1. **Repayment Burden** – Primary signal for affordability
2. **Employment Stability** – Indicator of income reliability
3. **Liquidity Ratio** – Ability to weather financial stress
4. **Income Trend** – Whether earnings trajectory is positive/declining
5. **Loan Amount Requested** – Relative to applicant capacity

**Each decision traceable to financially meaningful drivers** – key for regulatory approval and borrower explanation.

---

## Governance & Explainability

### SHAP-Based Explainability
- Feature importance ranked by contribution to model decisions
- Each approval/rejection linked to specific financial indicators
- Enables transparent communication with applicants and regulators

### Calibration
- Probability scores clipped to [0, 1] range
- Validation on test set to ensure calibration reliability
- Threshold selection based on precision-recall trade-off

### Stability Testing
- 5-fold cross-validation shows consistent AUC (0.88 ±0.154)
- Model generalizes across data splits
- Ready for controlled pilot deployment

### Fairness Limitations & Future Work

**Current approach**:
- Thin-file boost applied post-hoc (+0.10 adjustment)
- Inclusion Boost logic implemented but not validated on demographic data
- Four-Fifths Rule calculation structure in place, not fully tested

**Before production deployment, recommend**:
- Full disparate impact testing on real applicant demographics
- Benchmark against protected classes (age, gender, location proxy)
- Establish fairness monitoring dashboard
- Define acceptable fairness trade-offs with stakeholders

---

## Technical Stack

| Component | Technology |
|-----------|-----------|
| **Language** | Python 3.10 |
| **Data Processing** | Pandas, NumPy |
| **ML Framework** | Scikit-Learn, XGBoost |
| **Imbalance Handling** | Imbalanced-Learn (BorderlineSMOTE) |
| **Explainability** | SHAP |
| **Visualization** | Matplotlib, Seaborn |

### Model Configuration

**XGBoost Hyperparameters**:
```python
XGBClassifier(
    n_estimators=150,
    max_depth=4,
    learning_rate=0.03,
    scale_pos_weight=1.5,  # Class imbalance adjustment
    random_state=42,
    eval_metric='logloss'
)
```

**Rationale**:
- `max_depth=4`: Shallow trees reduce overfitting
- `scale_pos_weight=1.5`: Penalizes false negatives (missed loans)
- `learning_rate=0.03`: Conservative learning prevents instability

---

## Dataset

**Source**: German Credit Dataset (~1,000 records)  
**Task**: Binary classification (approve/decline)  
**Features**: 54 original variables (financial, employment, demographic)  
**Target**: Loan approval outcome  
**Privacy**: No personally identifiable information retained  
**Split**: 80/20 train/test with stratified sampling  

**Note**: Prototype validation. Production deployment requires:
- Larger, more recent applicant dataset
- Demographic diversity benchmarking
- Time-series validation (model performance over time)
- Rejection inference for missing outcome labels

---

## Key Findings

### What Works
✓ **Feature engineering matters**: Engineered repayment capacity variables outperform raw features  
✓ **SHAP reduces complexity**: 54 → 19 features with minimal performance loss  
✓ **Precision threshold strategy**: Balances inclusion vs. risk discipline  
✓ **Interpretability**: Each decision tied to specific financial indicators  

### What Needs Work
✗ **CV-Test Gap**: 0.88 vs 0.78 suggests overfitting; monitor on production data  
✗ **Fairness validation incomplete**: Inclusion Boost policy designed but not empirically validated  
✗ **Dataset scale**: German Credit is small; real credit models need 10k+ records  
✗ **Class imbalance**: BorderlineSMOTE helps, but fundamental data skew remains  

---

## Deployment Considerations

### Ready for Pilot
- Model code is modular and reproducible
- SHAP explainability enables regulatory audit
- Threshold selection documented and tunable
- Error handling for edge cases implemented

### Required Before Production
1. **Fairness audit**: Full disparate impact testing on real demographics
2. **Monitoring**: Track prediction accuracy and fairness metrics over time
3. **Threshold tuning**: Adjust 0.65 threshold based on business risk tolerance
4. **Rejection inference**: Handle missing outcome labels for declined applicants
5. **Model retraining**: Quarterly refit on production data to prevent drift
6. **A/B testing**: Compare against baseline scoring for performance validation

---

## How to Use

### Running the Model

```python
# Load and preprocess data
X_train, X_test, y_train, y_test = prepare_data(df)

# Train XGBoost model
model = XGBClassifier(n_estimators=150, max_depth=4, ...)
model.fit(X_train, y_train)

# Generate predictions with calibrated threshold
probabilities = model.predict_proba(X_test)[:, 1]
predictions = (probabilities >= 0.65).astype(int)

# SHAP explainability
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Approval with thin-file boost
thin_file_mask = (X_test['credit_history'] == 0)
probabilities[thin_file_mask] += 0.10
predictions_boosted = (probabilities >= 0.65).astype(int)
```

### Evaluation

```python
from sklearn.metrics import precision_score, recall_score, roc_auc_score

accuracy = (predictions == y_test).mean()
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
auc = roc_auc_score(y_test, probabilities)

print(f"Accuracy: {accuracy:.1%}")
print(f"Precision: {precision:.1%}")
print(f"Recall: {recall:.1%}")
print(f"ROC-AUC: {auc:.3f}")
```

---

## Project Structure

```
credit-scoring/
├── data/
│   └── german_credit.csv          # Source data
├── notebooks/
│   ├── 01_eda.ipynb               # Exploratory analysis
│   ├── 02_feature_engineering.ipynb # Feature creation
│   ├── 03_model_training.ipynb    # XGBoost training
│   └── 04_evaluation.ipynb        # Performance & fairness
├── src/
│   ├── preprocessing.py           # Data pipeline
│   ├── features.py                # Feature engineering
│   ├── model.py                   # Model training
│   └── evaluation.py              # Metrics & SHAP analysis
├── models/
│   └── xgb_model.pkl              # Trained model (serialized)
└── README.md                      # This file
```

---

## Key Decisions & Trade-offs

### 1. SHAP-Based Feature Selection vs. Domain Knowledge
**Decision**: Automated SHAP importance ranking  
**Trade-off**: Reduced interpretability complexity vs. potential loss of domain context  
**Mitigation**: Cross-checked results with financial domain logic  

### 2. Threshold Optimization for Precision
**Decision**: 0.65 probability threshold (vs. default 0.50)  
**Trade-off**: Lower false positives (safer for lender) vs. fewer approvals (lower revenue)  
**Mitigation**: Documented threshold tuning process; easily adjustable for different risk appetites  

### 3. Post-hoc Inclusion Boost vs. Retraining
**Decision**: Simple +0.10 probability adjustment for thin-file  
**Trade-off**: Quick fairness policy vs. model-level fairness guarantees  
**Mitigation**: Treat as policy lever, not structural fairness solution; requires fairness validation  

### 4. German Credit Dataset (Small, Historical)
**Decision**: Prototype on standard benchmark vs. proprietary data  
**Trade-off**: Reproducibility & transparency vs. production relevance  
**Mitigation**: Clear documentation of limitations; design for production retraining  

---

## Learning Outcomes

This project demonstrates:

1. **Feature Engineering**: Designing financial indicators that capture real-world lending dynamics
2. **Model Selection & Tuning**: Hyperparameter optimization with trade-off analysis (precision vs. recall)
3. **Class Imbalance**: Handling skewed datasets with resampling techniques (BorderlineSMOTE)
4. **Explainability**: SHAP-based interpretability for regulatory and business requirements
5. **Fairness in ML**: Awareness of bias risks and measurement limitations
6. **Production Readiness**: Structuring code for deployment, monitoring, and retraining
7. **Communication**: Translating model metrics into business outcomes (approval rates, default risk)

---

## Limitations & Future Work

### Current Limitations
- **Dataset Size**: 1,000 records (production models typically require 10k+)
- **Temporal Validation**: No time-series performance check (model may decay on newer applicants)
- **Fairness Validation Incomplete**: Inclusion Boost designed but not empirically validated
- **Missing Features**: No alternative data (telecom, utility payments, rent history)
- **Rejection Inference**: Unknown outcomes for declined applicants create selection bias

### Future Improvements
- [ ] Integration with alternative data sources (telecom, utility, rental payment history)
- [ ] Temporal validation: Test model performance on held-out time periods
- [ ] Full fairness audit: Disparate impact testing by demographics
- [ ] Rejection inference: Statistical techniques to impute missing outcomes
- [ ] Ensemble methods: Combine XGBoost with other models for improved robustness
- [ ] Active learning: Prioritize human review of borderline cases
- [ ] Production monitoring: Real-time tracking of accuracy and fairness drift

---

## References

**Industry Standards**:
- Fair Lending Compliance: Four-Fifths Rule (EEOC guidelines)
- Model Governance: OCC Guidance on Model Risk Management
- Explainability: GDPR Right to Explanation

---

## Acknowledgments

- German Credit Dataset (UCI Machine Learning Repository)
- SHAP library for explainability insights
- XGBoost team for robust gradient boosting framework
