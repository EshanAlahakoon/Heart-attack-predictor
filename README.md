# 🫀 Heart Attack Risk Predictor

## Overview
This project is a complete, end-to-end Machine Learning application designed to predict a patient's risk of experiencing a heart attack based on their medical vitals and lifestyle habits. It demonstrates a robust MLOps approach, moving from raw data preprocessing in a Jupyter Notebook to a fully deployed interactive web application.

## Key Features
* **Custom Scikit-Learn Pipeline:** Built a robust `Pipeline` featuring a custom `BaseEstimator` to safely parse string-based blood pressure data, ensuring seamless production inference.
* **Automated Preprocessing:** Utilized `ColumnTransformer` with `StandardScaler` for numerical data and `OneHotEncoder` (handling unknown variables safely) for categorical data.
* **Hyperparameter Tuning:** Fine-tuned a `RandomForestClassifier` using `GridSearchCV` to find the optimal model architecture.
* **Interactive UI:** Deployed the saved `.joblib` model into a user-friendly frontend using **Streamlit**, allowing users to input real-time patient data and receive instant risk probabilities.

## Tech Stack
* **Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn, Joblib
* **Web Framework:** Streamlit
* **Development Environment:** Google Colab, Local Virtual Environment (venv)

## How to Run Locally
1. Clone this repository to your local machine.
2. Ensure your virtual environment is activated.
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
