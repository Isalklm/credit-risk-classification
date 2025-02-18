# Credit Risk Analysis Report

## Overview of the Analysis
This analysis evaluates the effectiveness of a **Logistic Regression** model in predicting loan risk using historical lending data. The dataset includes **financial attributes of borrowers**, and the goal is to classify loans as either **healthy (0)** or **high-risk (1)**.

### **Purpose of the Analysis**
The purpose of this analysis is to determine whether a machine learning model can accurately assess **credit risk** based on borrower financial data. By training a classification model, we aim to help lenders **identify high-risk loans** before approval.

### **Financial Information in the Data**
The dataset contains **financial attributes of borrowers** that influence loan risk assessment, including:

- **Loan size** – The amount borrowed.
- **Interest rate** – The rate charged on the loan.
- **Borrower income** – The annual income of the borrower.
- **Debt-to-income ratio** – The proportion of income spent on debt.
- **Number of accounts** – The total active accounts held by the borrower.
- **Derogatory marks** – Negative records such as late payments.
- **Total debt** – The total debt owed by the borrower.
- **Loan status (`loan_status`)** – The **target variable**, indicating the loan's risk level:
  - **0** → Healthy loan (low risk of default).
  - **1** → High-risk loan (likely to default).

### **Machine Learning Process**
The analysis follows these key steps:
1. **Load and preprocess data** – Read `lending_data.csv` into a Pandas DataFrame.
2. **Feature engineering** – Define `X` (features) and `y` (target variable).
3. **Train-test split** – Split the dataset into 75% training and 25% testing sets.
4. **Train the model** – Fit a **Logistic Regression model** to the training data.
5. **Evaluate performance** – Measure model accuracy, precision, recall, and F1-score.

### **Machine Learning Method Used: Logistic Regression**
We applied a **Logistic Regression model** because:
- It is widely used for **binary classification problems**.
- It calculates the **probability of a loan being high-risk (1)** based on borrower attributes.
- It is **efficient and interpretable**, making it ideal for risk assessment.

The model was trained using:
- **Solver:** `lbfgs` (default for Logistic Regression).
- **Max Iterations:** `200` (ensuring full convergence).
- **Random State:** `1` (for reproducibility).

---

## Results
Below are the **performance metrics** for the Logistic Regression model:

### **Machine Learning Model: Logistic Regression**
- **Accuracy:** **99%**
- **Healthy Loans (0):**
  - Precision: **1.00** → Almost all predicted "healthy loans" are correct.
  - Recall: **0.99** → The model correctly identifies 99% of actual healthy loans.
  - **F1-score:** **1.00** → A perfect balance of precision and recall.
- **High-Risk Loans (1):**
  - Precision: **0.84** → 84% of predicted "high-risk loans" are actually high-risk.
  - Recall: **0.94** → The model correctly identifies 94% of actual high-risk loans.
  - **F1-score:** **0.89** → Strong performance in detecting high-risk loans.

---

## Summary
- The **Logistic Regression model performs exceptionally well** for predicting **healthy loans (0)**, with an **F1-score of 1.00**, meaning it rarely misclassifies them.
- The model **also performs well at detecting high-risk loans (1)**, with an **F1-score of 0.89**. However, it **misclassifies some high-risk loans**.
- **Recommendation:**
  - This model is **suitable for credit risk assessment** due to its **high accuracy (99%)**.
  - To **improve high-risk loan detection**, we could:
    - **Balance the dataset** (since high-risk loans are underrepresented).
    - **Use a different algorithm** (Random Forest, Neural Networks).
    - **Adjust the classification threshold** for better recall.

### **Final Verdict:** ✅ **Recommended for credit risk assessment, but could be improved further for high-risk loans.**

---

## 📌 Attribution and Code Sources
This project was developed as part of a bootcamp program. **Class materials, documentation, and external resources** were used to complete this challenge. The dataset and methodologies integrate:

- **Machine learning techniques from Scikit-Learn**
- **Data processing using Pandas and NumPy**
- **Model evaluation using Scikit-Learn metrics (Confusion Matrix, Classification Report)**
