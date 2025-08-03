# 🤖 Customer Churn Prediction – Telecom X

This project is part of the second phase of the **Telecom X** challenge, focusing on building a predictive model to anticipate customer churn. Using supervised machine learning techniques, the objective is to identify which customers are most likely to cancel their services, allowing for strategic retention actions.

## 🔍 About the Project

After completing the exploratory data analysis, this second part of the project focuses on:

- Building a complete machine learning pipeline
- Handling missing data, encoding categorical variables, and feature scaling
- Applying class balancing techniques (SMOTE)
- Training and tuning classification models (KNN and Random Forest)
- Evaluating model performance using metrics such as accuracy, precision, recall, and F1 Score
- Analyzing feature importance to understand key churn drivers

## 🛠️ Technologies and Libraries Used

- Python 🐍
- Pandas
- NumPy
- Scikit-learn
- Matplotlib & Seaborn
- Plotly
- imbalanced-learn (SMOTE)
- Yellowbrick
- Jupyter Notebook

## 🧠 Machine Learning Approach

### Data Preparation
- Null values were handled using the mean for numerical columns.
- Binary categorical variables were encoded manually.
- Multi-class categorical features were encoded using OneHotEncoder.
- A `MinMaxScaler` was used to normalize numerical values.
- Two new features were engineered to represent whether the customer had internet or phone service.

### Class Balancing
- SMOTE (Synthetic Minority Oversampling Technique) was applied to the training set to balance the target classes and reduce bias toward the majority class.

### Models Tested

#### K-Nearest Neighbors (KNN)
- Parameter tuning performed via `GridSearchCV`.
- Results showed limited recall and F1 Score, making it less suitable for the churn problem.

#### Random Forest Classifier 🌳
- Outperformed KNN in all metrics.
- Feature importance analysis helped reduce dimensionality to the top 17 variables.
- Hyperparameter tuning (e.g., depth, number of trees, class weights) was done to improve recall.
- Final model achieved strong recall and balanced performance.

## 📈 Model Performance (Random Forest)

| Metric     | Value    |
|------------|----------|
| Accuracy   | ~80%     |
| Precision  | ~66%     |
| Recall     | ~72%     |
| F1 Score   | ~69%     |

