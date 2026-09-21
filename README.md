# 📊 Website Traffic Analysis & Conversion Rate Prediction

> Predicting website conversion rates from visitor behavior using regression modeling and exploratory data analysis.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Wrangling-150458?style=flat&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Modeling-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-006400?style=flat)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat)
![License](https://img.shields.io/badge/License-MIT-blue?style=flat)

---

## 🧭 Overview

Every click, scroll, and session on a website tells a story about whether a visitor is going to convert. This project digs into a website traffic dataset to answer one core question:

> **Can we predict a visitor's conversion rate from their on-site behavior?**

The notebook walks through the full analytics pipeline — cleaning messy traffic data, visualizing behavior patterns, benchmarking six regression models, and packaging the best one into a reusable ML pipeline.

---

## ❓ Key Question Answered

**Which models best predict website conversion behavior, and why?**

The **Random Forest Regressor** and **Gradient Boosting Regressor** came out on top, both comfortably beating linear models, SVR, and XGBoost at capturing the non-linear relationships between visitor engagement and conversion.

---

## 🗂️ Dataset

| | |
|---|---|
| **Source** | [Website Traffic dataset (Kaggle)](https://www.kaggle.com/) via `kagglehub` |
| **Target variable** | `Conversion Rate` |
| **Features** | Page Views, Session Duration, Bounce Rate, Time on Page, Previous Visits, Traffic Source |
| **Type** | Tabular, mixed numeric + categorical |

---

## 🔬 Workflow


### 🧹 Preprocessing highlights
- Checked and handled missing values, NaNs, and duplicate rows
- Rescaled `Conversion Rate` to a 0–1 range and jittered the 1.0 ceiling to give the model learnable variance
- Balanced the target distribution with both **oversampling** and **undersampling** experiments
- Encoded `Traffic Source` with label/one-hot encoding, scaled numeric features with `StandardScaler`

### 📈 Exploratory analysis
- Distribution plots for Page Views and Session Duration
- Traffic Source breakdown
- Session Duration vs. Conversion Rate scatter analysis
- Correlation heatmap across all numeric features

---

## 🤖 Model Benchmark

Six regressors were trained and compared on held-out test data:

| Model | R² | RMSE | MAE | MAPE (%) | Remarks |
|---|---:|---:|---:|---:|---|
| Ridge Regression | 0.1094 | 0.0005 | 0.0003 | 3.54 | Regularized linear baseline |
| Lasso Regression | -0.0073 | 0.0006 | 0.0003 | 3.41 | Feature selection, poor generalization |
| **Random Forest** | **0.1845** | **0.0005** | **0.0002** | **2.50** | 🏆 Best MAPE, strong non-linear fit |
| **Gradient Boosting** | **0.1988** | 0.0005 | 0.0002 | 2.58 | 🏆 Best R², sequential learning |
| XGBoost | 0.1600 | 0.0005 | 0.0002 | 2.56 | Solid, room for tuning |
| Support Vector Regressor | -31.58 | 0.0032 | 0.0032 | 31.74 | Highly scale-sensitive, underperformed |

**Winner:** The final production pipeline wraps a `ColumnTransformer` (scaling + one-hot encoding) around a **Random Forest Regressor**.

---

## 🌲 Feature Importance

Across tree-based models, the strongest predictors of conversion were consistently:

1. **Session Duration**
2. **Page Views**
3. **Previous Visits**

This points to visitor *engagement intensity* — not just traffic volume — as the real driver of conversions.

---

## 📐 Model Evaluation

Beyond R²/RMSE/MAE/MAPE, the notebook validates the Random Forest model with:

- ✅ 5-fold cross-validation
- ✅ Adjusted R² (penalizing feature count)
- ✅ Residual distribution & normality (Shapiro-Wilk test)
- ✅ Actual vs. Predicted plot
- ✅ Learning curve (train size vs. R²) to check for over/underfitting

---

## 🛠️ Tech Stack

`Python` · `Pandas` · `NumPy` · `Matplotlib` · `Seaborn` · `scikit-learn` · `XGBoost` · `SciPy` · `KaggleHub`

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost scipy kagglehub
```

### Run the notebook
```bash
jupyter notebook website-traffic-analysis_Data_Analyst.ipynb
```

### Use the saved model
```python
import pickle

with open("RandomForest_pipeline_model.pkl", "rb") as f:
    model = pickle.load(f)

prediction = model.predict(new_data)
```

---

## 💡 Recommendations & Next Steps

- 🔧 **Hyperparameter tuning** with GridSearchCV / RandomizedSearchCV on the tree-based models
- 📈 **Feature expansion**: campaign indicators, external events, hourly/session-based aggregations
- 🚀 **Deployment**: plug the pipeline into a real-time dashboard or traffic-alerting system
- 🔁 **Continuous retraining** as new traffic logs arrive, to adapt to shifting visitor behavior

---

## 📁 Project Structure


---

## 👤 Author

**Shahnawaj Siddique**
Aspiring Data & Business Analyst | SQL · Power BI · Tableau · Python
📍 Indore, Madhya Pradesh, India

---

## 📄 License

This project is available under the MIT License.
