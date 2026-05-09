# 🛒 E-commerce Return Prediction

![Python](https://img.shields.io/badge/Python-3.11-blue)
![sklearn](https://img.shields.io/badge/scikit--learn-1.3.2-orange)
![Status](https://img.shields.io/badge/Status-Concluded-red)

---

## Problem Statement
Predict whether an e-commerce order will be returned using customer behaviour, order details, and platform data — enabling businesses to proactively reduce reverse logistics costs.

---

## Dataset
| Property | Value |
|---|---|
| Source | Kaggle — Hamna Munir |
| Type | Synthetic / Simulated |
| Records | 10,000 orders |
| Features | 38 columns |
| Years | 2021–2024 |
| Return Rate | 6.4% |
| Null Values | None |

---

## Methodology

### 1. Exploratory Data Analysis
- Analysed return rates across product categories, platforms,customer segments, payment methods and devices
- Built correlation heatmap across all numerical features
- Identified `product_rating` as the only feature with moderate correlation (-0.49) with returns
- Found very uniform return rates (5–8%) across all groups

### 2. Feature Engineering
- Removed leaky post-purchase columns (return_reason, customer_satisfaction, product_rating)
- Applied Label Encoding for tree and linear models
- Stratified 80/20 train/test split
- Applied SMOTE for class imbalance handling

### 3. Model Training
Compared 5 models × 2 balancing techniques = 10 experiments

| Model | Technique | Balanced Accuracy | Recall | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | Class Weights | 53.96% | 53.91% | 53.07% |
| Logistic Regression | SMOTE | 50.38% | 24.22% | 52.34% |
| Decision Tree | Class Weights | 51.40% | 9.38% | 51.40% |
| Random Forest | SMOTE | 50.59% | 3.91% | 51.08% |
| XGBoost | Class Weights | 50.32% | 2.34% | 49.01% |
| Random Forest | Class Weights | 50.00% | 0.00% | 51.74% |

### 4. Hyperparameter Tuning
- Applied RandomizedSearchCV with Stratified 5-Fold CV
- Optimised for Balanced Accuracy
- Tested 50 parameter combinations

**Tuning Result — No improvement:**
| Metric | Baseline | Tuned |
|---|---|---|
| Balanced Accuracy | 53.96% | 53.35% |
| Recall | 53.91% | 51.56% |
| ROC-AUC | 53.07% | 53.03% |

---

## 📉 Conclusion

The best model achieved **53.96% Balanced Accuracy** — only 
marginally above random guessing (50%).

### Why Performance Was Low

- Synthetic dataset with artificially uniform distributions
- Return rates vary only 5–8% across ALL feature groups
- No single strong predictor — product_rating removed
due to data leakage (only available post-purchase)
Hyperparameter tuning confirmed model was already
at its ceiling with default parameters

### Key Learning
> Identifying when data does not support a hypothesis 
> is as valuable as building a high-performing model. 
> This project demonstrates honest scientific thinking 
> over metric chasing.

---

## 🛠️ Tech Stack
| Tool | Purpose |
|---|---|
| Python 3.11 | Core language |
| Pandas, NumPy | Data processing |
| Matplotlib, Seaborn | Visualisation |
| Scikit-learn | ML models |
| Imbalanced-learn | SMOTE |
| XGBoost | Gradient boosting |
| Jupyter | Experimentation |


---

# How to Run
```bash
# Clone repository
git clone https://github.com/shamilikolli/ecommerce-return-prediction
cd ecommerce-return-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download dataset from Kaggle
# Place in data/ folder as ecommerce_sales_customer_behavior.csv

# Run notebooks in order
jupyter notebook notebooks/01_EDA.ipynb
```

---

## Skills Demonstrated
- End to end ML project structure
- Thorough EDA with business context
- Feature engineering and leakage prevention
- Class imbalance handling (SMOTE vs Class Weights)
- Multi-model comparison framework
- Hyperparameter tuning with Cross Validation
- Result interpretation and documentation