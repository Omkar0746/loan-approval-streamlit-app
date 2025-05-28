import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

# Load dataset
df = pd.read_csv('Credit_Risk_Dataset_with_Loan_Status.csv')

# Features and target
X = df.drop(['loan_status', 'loan_approved', 'dlq_2yrs'], axis=1)
y = df['loan_approved']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'model.pkl')
print("✅ Model trained and saved as model.pkl")
