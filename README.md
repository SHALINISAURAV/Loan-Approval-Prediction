# 🚀 Loan Approval Prediction  

A **machine learning project** to predict **loan approval status** based on applicant details. This system helps automate the loan approval process, making it **faster, more efficient, and data-driven**. 🤖  

---

## 📝 Project Overview  

This project implements a **complete ML pipeline** for predicting whether a loan will be approved or rejected.  

**Workflow:**  
1. **Data Preprocessing** 🧹  
   - Handle missing values  
   - Encode categorical variables  
   - Scale numerical features  

2. **Model Training** 🌲🌳  
   - **Decision Tree Classifier**  
   - **Random Forest Classifier** (best performer)  

3. **Model Evaluation** ✅  
   - Accuracy Score  
   - Classification Report  
   - Confusion Matrix  

4. **Model Persistence** 💾  
   - Save the best-performing model (`loan_model.pkl`) for future predictions.  

---

## 🛠️ Technologies & Libraries  

| Library        | Purpose |
|----------------|---------|
| **pandas** 📊  | Data manipulation and analysis |
| **NumPy**      | Numerical operations |
| **Matplotlib & Seaborn** 📈 | Data visualization |
| **Scikit-learn** | Machine learning models and utilities |
| &nbsp; ├─ `train_test_split` | Split data into training/testing |
| &nbsp; ├─ `LabelEncoder` | Encode categorical variables |
| &nbsp; ├─ `StandardScaler` | Scale features |
| &nbsp; ├─ `DecisionTreeClassifier` | Classification model |
| &nbsp; ├─ `RandomForestClassifier` 🎉 | Ensemble model |
| &nbsp; ├─ `classification_report`, `accuracy_score`, `confusion_matrix` | Model evaluation |
| **pickle** 💾 | Save/load trained models |

---

## ⚙️ Installation & Setup  

1️⃣ **Clone the repository**  
```bash
git clone <https://github.com/SHALINISAURAV/Loan-Approval-Prediction>
```

2️⃣ **Install dependencies**  
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

3️⃣ **Run the Jupyter Notebook**  
Open `Loan Approval Prediction.ipynb` in Jupyter Notebook/Lab to execute and explore the project.  

---

## 📊 Dataset  

**File:** `loan_approval_dataset.csv`  
Contains details of loan applicants with the following features:  

| Feature | Description |
|---------|-------------|
| `loan_id` | Loan ID |
| `no_of_dependents` | Number of dependents |
| `education` 🎓 | Education level |
| `self_employed` | Employment type |
| `income_annum` 💰 | Annual income |
| `loan_amount` | Loan amount requested |
| `loan_term` | Loan term (years) |
| `cibil_score` 💯 | CIBIL credit score |
| `residential_assets_value` 🏡 | Residential assets value |
| `commercial_assets_value` 🏢 | Commercial assets value |
| `luxury_assets_value` 💎 | Luxury assets value |
| `bank_asset_value` | Bank asset value |
| `loan_status` | **Target variable** (Approved / Rejected) |

---

## 💡 Usage Example  

```python
import pickle
import pandas as pd

# Load the saved model
with open('loan_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Example new applicant data
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

# Preprocess new_df using the same LabelEncoder & StandardScaler as training
# ... (preprocessing steps here)

# Predict loan status
prediction = model.predict(new_df)

if prediction[0] == 1:
    print("✅ Loan Approved")
else:
    print("❌ Loan Rejected")
```

---

## 👩‍💻 Author  
**Shalini Saurav**  

---

## 📌 Key Highlights  
- **Random Forest Classifier** achieved the best accuracy  
- **Reusable trained model** stored as `.pkl` file  
- Complete end-to-end **loan approval prediction pipeline**  
