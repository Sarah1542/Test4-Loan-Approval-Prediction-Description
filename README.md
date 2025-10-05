# 🏦 Loan Approval Prediction

## Project Overview

This is a **Binary Classification** project aimed at building robust machine learning models capable of accurately predicting a loan application's outcome (Approved/Rejected) based on the applicant's financial and demographic profiles. The methodology specifically addresses the challenge of **class imbalance** using the **SMOTE** technique to ensure fair training across both outcome categories.

---

## ⚙️ Technologies and Libraries Used

* **Python 3.x**
* **Pandas & NumPy:** For data manipulation, handling, and mathematical operations.
* **Matplotlib & Seaborn:** For data visualization and understanding feature distribution.
* **Scikit-learn (sklearn):** For core algorithm implementation and model evaluation.
* **Imblearn (imbalanced-learn):** For specialized data balancing techniques.

---

## 📁 Dataset Details

The dataset contains applicant features, including **Credit History** (the most influential factor), Applicant Income, Loan Amount, marital status, and gender.

| Feature | Type | Description |
| :--- | :--- | :--- |
| **Credit History** | Categorical | The applicant's credit reliability (Crucial predictor). |
| **Applicant Income** | Numerical | The income of the person applying for the loan. |
| **Loan Status** | Target | The predicted outcome (Y = Approved, N = Rejected). |

---

## 🔧 Steps Performed (Advanced Methodology)

1.  **Data Exploration & Missing Value Imputation** 🧹
    * Visualized data distributions and feature correlations.
    * Missing values were handled using appropriate imputation strategies (mean, mode).

2.  **Categorical Encoding** 🏷️
    * Categorical features (e.g., Gender, Marital Status) were converted into numerical format using methods like **LabelEncoder**.

3.  **Handling Class Imbalance** ⚖️
    * The **SMOTE (Synthetic Minority Over-sampling Technique)** method was applied to **generate synthetic samples** for the minority class. This prevents the model from being biased toward predicting 'Approval' (the majority class).

4.  **Data Scaling** 📏
    * **StandardScaler** was used to normalize quantitative features, ensuring all variables contribute equally to the model training process.

5.  **Model Training and Comparison** 🧠
    * Two powerful binary classification models were trained and compared:
        1.  **Logistic Regression**
        2.  **Decision Tree Classifier**

6.  **Comprehensive Evaluation** ⭐
    * Model performance was evaluated using **Confusion Matrix** and **Classification Report** to accurately assess prediction quality for both the 'Approved' and 'Rejected' outcomes.

---

## 📦 Libraries Used

```bash
pandas
numpy
matplotlib.pyplot
seaborn
sklearn.preprocessing (StandardScaler, LabelEncoder)
sklearn.model_selection (train_test_split)
imblearn.over_sampling (SMOTE)
sklearn.linear_model (LogisticRegression)
sklearn.tree (DecisionTreeClassifier)
sklearn.metrics (classification_report, confusion_matrix)
