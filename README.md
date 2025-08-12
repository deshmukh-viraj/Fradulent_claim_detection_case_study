# Fraudulent Claim Detection

## 📌 Overview
This project aims to detect fraudulent insurance claims using a **machine learning classification approach**.  
Global Insure, a leading insurance company, faces significant losses due to fraudulent claim payouts. Manually inspecting thousands of claims is inefficient, so we use **data-driven predictive modeling** to automatically classify claims as 'Fraudulent' or 'Legitimate'.  

By implementing this model, the company can:
- Flag suspicious claims for review earlier in the process.
- Reduce financial losses due to fraudulent payouts.
- Improve claims handling efficiency.

---

## 📝 Problem Statement
Global Insure processes thousands of claims annually, but a portion turns out to be fraudulent.  
The **objective** is to leverage **historical claim and customer data** to predict whether a new claim is fraudulent before payout approval.

Key questions addressed:
1. How can we analyse historical data to detect fraudulent claim patterns?
2. Which claim/customer features are most predictive?
3. Can we reliably classify new claims as fraudulent?
4. What policy improvements can we make from the insights?

---

## 📊 Dataset & Features
- **Rows:** 1,000 claims  
- **Columns:** 40  
- **Target Variable:** `fraud_reported` (Y = Fraudulent, N = Legitimate)

### Sample Columns
| Column Name | Description |
|-------------|-------------|
| months_as_customer | Duration in months with insurer |
| age | Age of insured |
| policy_csl | Combined single limit |
| incident_type | Type/category of incident |
| incident_severity | Level of damage |
| property_damage | YES/NO |
| total_claim_amount | Total claimed amount |
| injury_claim | Amount claimed for injuries |
| vehicle_claim | Amount claimed for vehicles |
| auto_make, auto_model, auto_year | Vehicle details |
| fraud_reported | Target variable |

---

## ⚙️ Project Workflow
1. **Data Preparation:** Load data, handle missing values, format dates.  
2. **Data Cleaning:** Remove redundant features, fix inconsistencies.  
3. **Train-Test Split:** 70-30 split.  
4. **EDA:** Explore target distribution, correlations, and categorical feature impact.  
5. **Feature Engineering:** One-hot encoding, scaling, derived features.  
6. **Model Building:** Tested Logistic Regression, Decision Tree, Random Forest, XGBoost.  
7. **Evaluation:** Compare models by Accuracy, Precision, Recall, F1-Score.

---

## 📈 EDA Highlights

### Fraud Distribution
The dataset is imbalanced:
- **Fraudulent claims:** ~25%
- **Legitimate claims:** ~75%

<Figure size 640x480 with 1 Axes><img width="581" height="458" alt="image" src="https://github.com/user-attachments/assets/d1b5320f-1871-4d81-85a7-b549b736939b" />


---

### Key Numerical Insights
| Feature                  | Fraudulent Claims (Avg) | Legitimate Claims (Avg) |
|--------------------------|------------------------|--------------------------|
| `total_claim_amount`     | Higher                 | Lower                    |
| `injury_claim`           | Higher                 | Lower                    |
| `vehicle_claim`          | Higher                 | Lower                    |
| `months_as_customer`     | Slightly lower         | Higher                   |
| `incident_hour_of_the_day`| Peaks in late evenings | Less distinct pattern    |

---

### High-Impact Features (Top 10 from Random Forest)
1. total_claim_amount
2. injury_claim
3. vehicle_claim
4. incident_severity
5. policy_annual_premium
6. months_as_customer
7. capital-gains
8. capital-loss
9. property_damage
10. auto_year

![Feature Importance Plot](images/feature_importance.png)

---

### Correlation Heatmap
A heatmap revealed strong correlation among `total_claim_amount`, `injury_claim`, `vehicle_claim`, and partially with fraud.

![Correlation Heatmap](images/correlation_heatmap.png)

---

## 🛠️ Tech Stack
- Python 3.x
- **Data Processing:** pandas, numpy  
- **Visualization:** matplotlib, seaborn  
- **ML Models:** scikit-learn, xgboost

---

## 🚀 How to Run
