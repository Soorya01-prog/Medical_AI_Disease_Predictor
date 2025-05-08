# Install required packages
!pip install scikit-learn pandas matplotlib seaborn

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import joblib
from google.colab import files
from sklearn.preprocessing import LabelEncoder  # Import LabelEncoder
from sklearn.impute import SimpleImputer # Import SimpleImputer to fill NaN

# Step 1: Load the processed data
uploaded = files.upload()
data = pd.read_csv(next(iter(uploaded.keys())))

# ... (Rest of your data preprocessing code remains the same) ...


# ... (previous code) ...

# Step 4: Standardize the Features
# Create a SimpleImputer to replace NaNs with the mean
imputer = SimpleImputer(strategy='mean')  # You can use 'median' or other strategies as well

# Select only numerical features for imputation
numerical_features = X_train.select_dtypes(include=np.number).columns

# Fit the imputer on the numerical training data and transform both train and test data
X_train_sd = imputer.fit_transform(X_train[numerical_features])
X_test_sd = imputer.transform(X_test[numerical_features])

# Standardize the data after imputation
scaler = StandardScaler()
X_train_sd = scaler.fit_transform(X_train_sd)
X_test_sd = scaler.transform(X_test_sd)

# ... (Rest of your code for model training and evaluation)
# Step 5: Train Logistic Regression
print("\nTraining Logistic Regression...")
param_grid_lr = {'C': [0.1, 1, 10], 'penalty': ['l2'], 'solver': ['lbfgs', 'liblinear']}
lr_model = LogisticRegression()
grid_search_lr = GridSearchCV(lr_model, param_grid_lr, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
grid_search_lr.fit(X_train_sd, Y_train)

# Save the best Logistic Regression model
best_lr_model = grid_search_lr.best_estimator_
joblib.dump(best_lr_model, 'Logistic_Regression_best_model.joblib')
print("\nLogistic Regression best model saved as 'Logistic_Regression_best_model.joblib'")

# Evaluate Logistic Regression
y_pred_lr = best_lr_model.predict(X_test_sd)
print("\nLogistic Regression Classification Report:\n", classification_report(Y_test, y_pred_lr))

# Confusion Matrix for Logistic Regression
cm_lr = confusion_matrix(Y_test, y_pred_lr)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Step 6: Train Random Forest
print("\nTraining Random Forest...")
param_grid_rf = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}
rf_model = RandomForestClassifier()
grid_search_rf = GridSearchCV(rf_model, param_grid_rf, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
grid_search_rf.fit(X_train_sd, Y_train)

# Save the best Random Forest model
best_rf_model = grid_search_rf.best_estimator_
joblib.dump(best_rf_model, 'Random_Forest_best_model.joblib')
print("\nRandom Forest best model saved as 'Random_Forest_best_model.joblib'")

# Evaluate Random Forest
y_pred_rf = best_rf_model.predict(X_test_sd)
print("\nRandom Forest Classification Report:\n", classification_report(Y_test, y_pred_rf))

# Confusion Matrix for Random Forest
cm_rf = confusion_matrix(Y_test, y_pred_rf)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens')
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print("\nModel training and evaluation complete. Models saved as 'Logistic_Regression_best_model.joblib' and 'Random_Forest_best_model.joblib'.")

