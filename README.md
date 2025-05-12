# ❤️ Heart Disease Prediction using ML Models

This repository contains a Jupyter Notebook that implements three supervised machine learning models — **Logistic Regression**, **Random Forest**, and **Support Vector Classifier (SVC)** — to predict the likelihood of heart disease in individuals based on clinical and demographic attributes.

## 📁 File Description

- `logistic_randomforest_SVC.ipynb`: Contains the entire machine learning pipeline including data loading, preprocessing, training of models, evaluation, and results comparison.

## 🧠 Project Overview

Heart disease is one of the leading causes of death globally. Early prediction and detection can save lives. This notebook aims to:

- Predict heart disease using machine learning models.
- Compare the performance of Logistic Regression, Random Forest, and SVC.
- Use common evaluation metrics like Accuracy, Precision, Recall, F1 Score, and Confusion Matrix.

## 📊 Dataset

- **Source**: [heart-disease-dataset](https://www.kaggle.com/datasets/mirzahasnine/heart-disease-dataset)
- **Records**: 303 patient records
- **Features**: 13 clinical and demographic features including:
  - Age, sex, chest pain type, resting blood pressure, cholesterol, fasting bloo sugar, etc.
- **Target**: Presence or absence of heart disease (binary classification)

## 🛠️ Technologies Used

- Python 3.x
- Jupyter Notebook
- NumPy, Pandas
- Scikit-learn
- Matplotlib, Seaborn

## 🧪 Model Evaluation

| Model               | Accuracy | Precision | Recall | F1-Score |
|--------------------|----------|-----------|--------|----------|
| Logistic Regression| 85.25%   | 86%       | 84%    | 85%      |
| Random Forest      | 89.34%   | 90%       | 88%    | 89%      |
| SVC                | 87.12%   | 88%       | 86%    | 87%      |

> Confusion matrices and classification reports are included in the notebook for deeper analysis.

## 🚀 How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/heart-disease-prediction.git
   ```
2. Navigate into the project directory:
   ```bash
   cd heart-disease-prediction
3. Launch the Jupyter Notebook:
   ```bash
   jupyter notebook logistic_randomforest_SVC.ipynb
📌 Future Improvements
-Hyperparameter tuning with GridSearchCV
-Add more models like XGBoost, KNN
-Deploy as a web app with Streamlit or Flask

🤝 Contributions
Contributions are welcome! Please fork this repo and submit a pull request for improvements.
