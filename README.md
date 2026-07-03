# employee-attrition-prediction

# 📊 Employee Attrition Prediction

Predicting employee attrition using classification models on the IBM HR Analytics dataset — with a full pipeline from raw data to HR-ready business recommendations.

## 🎯 Overview

Every company loses employees — but losing the *wrong* employees at the *wrong* time is expensive. This project builds a machine learning system that predicts whether an employee is likely to leave, based on factors like overtime, job role, travel frequency, income, and tenure — then translates the results into insights an HR team could act on immediately.

**Dataset:** [IBM HR Analytics Employee Attrition](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) — 1,470 employees, 35 features.

---

## 🗂️ Repository Structure

```
├── analysis.ipynb                     # Full notebook — all tasks, code, and insights
├── HR_Attrition.csv                   # Dataset used
├── summary.docx                       # Non-technical summary for HR stakeholders
├── charts/
│   ├── chart1_attrition_dept_role.png # Attrition rate by Department & Job Role
│   ├── chart2_income_boxplot.png      # Monthly Income: Stayed vs Left
│   ├── chart3_confusion_matrix.png    # Confusion Matrix (best model)
│   ├── chart4_feature_importance.png  # Top 10 Feature Importances
│   └── chart5_roc_curve.png           # ROC Curve — all 3 models compared
└── README.md
```

---

## 🛠️ Tech Stack

|Tool                                   | Purpose |
|---|---|
| Python 3.x | Core language |
| Pandas / NumPy | Data loading, cleaning, manipulation |
| Scikit-learn | Preprocessing, modeling, evaluation |
| Matplotlib / Seaborn | Visualization |
| Google Colab | Development environment |

---

## 🔍 Approach

1. **Data Cleaning** — Dropped constant/irrelevant columns (`EmployeeNumber`, `Over18`, `StandardHours`, `EmployeeCount`), one-hot encoded 7 categorical features, scaled numeric features with `StandardScaler`.
2. **EDA** — Explored attrition patterns across department, job role, income, work-life balance, and tenure.
3. **Modeling** — Trained and compared 3 classifiers, handling class imbalance via `class_weight='balanced'` (Logistic Regression, Random Forest) and `sample_weight` (Gradient Boosting).
4. **Evaluation** — Compared models on Precision, Recall, F1, and ROC-AUC — prioritizing ROC-AUC and Recall over raw accuracy, since the dataset is imbalanced (~84% stayed vs ~16% left).
5. **Interpretation** — Extracted feature importances/coefficients to identify the strongest drivers of attrition.

---

## 🤖 Model Comparison

| Model | Precision (Leavers) | Recall (Leavers) | F1-Score | ROC-AUC |
|---|---|---|---|---|
| **Logistic Regression** ⭐ | 0.34 | **0.62** | 0.44 | **0.799** |
| Gradient Boosting | 0.43 | 0.43 | 0.43 | 0.778 |
| Random Forest | 0.57 | 0.09 | 0.15 | 0.772 |

**Best model: Logistic Regression** — despite lower raw accuracy, it catches significantly more actual leavers (62% recall) than Random Forest (9% recall), making it far more useful for a real HR early-warning use case where missing a leaver is costlier than a false alarm.

---

## 💡 Key Findings

- **Sales** has the highest departmental attrition (**20.6%**), followed by HR (19.0%); R&D is most stable (13.8%).
- **Sales Representatives** are the highest flight-risk role at **39.8%** attrition — more than double the next-highest role.
- Employees who left earned **~30% less on average** (₹4,787 vs ₹6,833 monthly) — but pay is not the strongest predictor.
- **Overtime, frequent travel, and job role** are stronger statistical predictors of attrition than salary alone.
- **Early tenure is the biggest risk window** — 36% of Year-0 employees and 34.5% of Year-1 employees leave; this drops sharply after Year 2.

**Top predictors (Logistic Regression coefficients):**
1. Job Role — Laboratory Technician
2. Overtime = Yes
3. Frequent Business Travel

---

## 📈 Visualizations

| | |
|---|---|
| Attrition by Dept/Role | Income: Stayed vs Left |
| Confusion Matrix | Top 10 Feature Importances |
| ROC Curve Comparison | |

*(See `/charts` for all rendered images.)*

---

## 📋 Business Recommendations

1. **Cap overtime / introduce comp-time policies** for Sales Representatives and Laboratory Technicians — the roles most affected by overtime-driven attrition.
2. **Build a structured first-year retention program** — proactive check-ins and career conversations during an employee's first 24 months, the highest-risk window for departure.

**Limitation:** The model achieves ~34% precision — roughly 2 in 3 employees flagged as "at risk" won't actually leave. It should guide *where* to focus retention conversations, not serve as a definitive prediction. It also can't capture factors outside the dataset, like manager relationships or competing offers.

---

## 🚀 Running This Project

```bash
git clone https://github.com/anyagupta03/employee-attrition-prediction.git
cd employee-attrition-prediction
jupyter notebook analysis.ipynb
```

Or open `analysis.ipynb` directly in [Google Colab](https://colab.research.google.com/).

**Requirements:** `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`

---

## 👩‍💻 Author

**Anya Gupta**
B.Tech AI & CS, Banasthali Vidyapith
[GitHub](https://github.com/anyagupta03) · [LinkedIn](#)
