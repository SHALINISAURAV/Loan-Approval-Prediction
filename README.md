🚀 Loan Approval Prediction

This project uses machine learning models to predict loan approval status based on applicant details. Our mission is to build and evaluate predictive models that can assist in automating the loan approval process, making it faster and more efficient. 🤖

📝 Project Description
This project is a complete pipeline for a machine learning solution. The workflow includes:

Data Preprocessing: Cleaning and preparing the dataset for model training. This includes handling missing values, encoding categorical variables, and scaling numerical features. 🧹

Model Training: Training two powerful classification models: a Decision Tree Classifier and a Random Forest Classifier, to predict the loan status (Approved or Rejected). 🌲🌳

Model Evaluation: Assessing the performance of both models using key metrics like accuracy, classification reports, and confusion matrices to find the most effective one. ✅

Model Persistence: Saving the best-performing model (the Random Forest model) as a pickle file (loan_model.pkl) so it can be used for future predictions without retraining. 💾

🛠️ Technologies and Libraries

The project is built with the following essential Python libraries:

pandas: For all your data manipulation and analysis needs. 📊
NumPy: The foundation for numerical operations.
Matplotlib & Seaborn: For creating stunning data visualizations. 📈
Scikit-learn: The machine learning powerhouse for:
train_test_split: Splitting data for training and testing.
LabelEncoder: Converting categorical labels into numbers.
StandardScaler: Scaling numerical features for better model performance.
DecisionTreeClassifier: Our first predictive model.
RandomForestClassifier: The ensemble model that won! 🎉
classification_report, accuracy_score, confusion_matrix: Evaluating model success.

📊 Dataset
The project uses the loan_approval_dataset.csv file. It's packed with key features about loan applicants, including:

loan_id
no_of_dependents
education 🎓
self_employed
income_annum 💰
loan_amount
loan_term
cibil_score 💯
residential_assets_value 🏡
commercial_assets_value 🏢
luxury_assets_value 💎
bank_asset_value

loan_status (Our target variable! Approved or Rejected)

💡 Usage
pickle: Saving and loading our trained model.

import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the saved model
with open('loan_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Example of a new data point
new_data = {
    'loan_id': [9999],
    'no_of_dependents': [2],
    'education': ['Graduate'],
    'self_employed': ['No'],
    'income_annum': [5000000],
    'loan_amount': [15000000],
    'loan_term': [10],
    'cibil_score': [750],
    'residential_assets_value': [5000000],
    'commercial_assets_value': [1000000],
    'luxury_assets_value': [12000000],
    'bank_asset_value': [2000000]
}

new_df = pd.DataFrame(new_data)

# Preprocess the new data in the same way as the training data
# Note: You'll need to use the same LabelEncoder and StandardScaler instances
# that were fitted during the training phase.
# ... (preprocessing steps)

# Make a prediction
prediction = model.predict(new_df)

if prediction[0] == 1:
    print("Loan Approved")
else:
    print("Loan Rejected")

👩‍💻 Author
  Shalini Saurav

