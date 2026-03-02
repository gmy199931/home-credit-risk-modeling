# Home Credit Default Risk — Loan Default Risk Modeling

## Overview
This project builds baseline models to predict loan default risk using the Kaggle *Home Credit Default Risk* dataset.  
The workflow covers business framing, exploratory analysis, feature engineering, model training, and model comparison.

**Goal:** Identify key factors associated with default risk and produce a reproducible, portfolio-ready analysis.

---

## Dataset
Source: Kaggle — Home Credit Default Risk  
Main files used:
- `application_train.csv` (training data)
- `application_test.csv` (test data; optional for future extension)
- `columns_description.csv` (data dictionary)

**Target definition**
- `TARGET = 1` → default
- `TARGET = 0` → non-default

**Class imbalance**
- Default rate is ~8%, so we evaluate models using **ROC-AUC** instead of accuracy.

---

## Project Structure
home-credit-default-risk/

├─ data/ # Kaggle CSVs (not uploaded to GitHub)

├─ draft/ # Previous uncleaned code

├─ notebooks/

│ ├─ 01_baseline_restructured.ipynb

│ ├─ 02_modeling_refined.ipynb

├─ outputs/ # exported predictions / dashboard-ready CSVs

└─ README.md

---

## Methodology

### 1. Exploratory Analysis (01_baseline_restructured.ipynb)
- Verified severe class imbalance (default rate ~8%)
- Tested initial hypotheses about key drivers:
  - Age (AGE_YEARS)
  - Income
  - Leverage (Credit-to-Income ratio)

**Key EDA findings**
- Higher age → lower default risk (univariate trend)
- Higher income → lower default risk
- Higher leverage → higher default risk

---

### 2. Modeling (02_modeling_refined.ipynb)

#### Logistic Regression (baseline)
- Started with simple features (age / income / leverage) → limited performance
- Adding external credit scores **EXT_SOURCE_1/2/3** significantly improved performance

**Result:** Logistic Regression achieved **AUC ≈ 0.75**

#### Interpretation: conditional effects & multicollinearity
- Age coefficient sign changed after adding EXT_SOURCE variables
- Correlation check showed `AGE_YEARS` is moderately correlated with `EXT_SOURCE_1` (~0.6)
- This indicates overlapping information and **conditional effects**, not a modeling error

#### LightGBM (tree-based baseline)
- Trained a LightGBM baseline using the same feature set
- Compared AUC and feature importance

**Result:** LightGBM achieved **AUC ≈ 0.74**, similar to Logistic Regression  
This suggests the feature space with a small set of high-information features is close to linearly separable.

#### Class imbalance experiment
- Tested `scale_pos_weight` to up-weight the minority class
- ROC-AUC remained stable since AUC is threshold-independent

---

## Key Conclusions
- External credit score features (**EXT_SOURCE**) are dominant predictors of default risk.
- Logistic Regression performed comparably to LightGBM on a small, high-information feature set.
- Feature quality had a larger impact than model complexity in this setup.
- ROC-AUC is appropriate for imbalanced classification; accuracy would be misleading.

---

## How to Run
1. Download the dataset from Kaggle and place CSV files in `data/`
2. Create a Python environment and install dependencies:
```bash

pip install pandas numpy scikit-learn matplotlib seaborn lightgbm
