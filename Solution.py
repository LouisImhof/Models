import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import SelectKBest, chi2, RFECV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib
matplotlib.use('TkAgg')
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


# Load dataset
df = pd.read_csv("/Users/louisimhof/Downloads/biased_leukemia_dataset.csv")

print(df.head())
print(df.isnull().sum())


def remove_outliers(df, column):
    # Calculate Q1 (25th percentile) and Q3 (75th percentile)
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    # Calculate IQR (Interquartile Range)
    IQR = Q3 - Q1
    # Define lower and upper bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    # Remove outliers
    df_no_outliers = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df_no_outliers


for col in df.select_dtypes(include=[np.number]).columns:
    df = remove_outliers(df, col)


# Remove outliers for specific columns
columns_to_check = ["WBC_Count", "RBC_Count", "Platelet_Count",
                    "BMI", "Hemoglobin_Level", "Bone_Marrow_Blasts"]

for column in columns_to_check:
    df = remove_outliers(df, column)

print(f"Data after outlier removal: {df.shape} rows, {df.shape[1]} columns")

# Check for duplicates
duplicates = df.duplicated().sum()
print(f"Number of duplicate rows: {duplicates}")

# Find categorical columns
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
print(f"Categorical columns: {categorical_cols}")

# Encode categorical columns
label_encoders = LabelEncoder()

for col in categorical_cols:
    if df[col].dtype == "object":
        df[col] = label_encoders.fit_transform(df[col])
        print(f"Encoded column: {col}")


print(df.head())


# Create Features
df['Hemoglobin_WBC'] = df['Hemoglobin_Level'] / df['WBC_Count']
df['Platelet_WBC'] = df['Platelet_Count'] / df['WBC_Count']
df['Platelet_RBC'] = df['Platelet_Count'] / df['RBC_Count']
df['Platelet_Hemoglobin'] = df['Platelet_Count'] / df['Hemoglobin_Level']
df['WBC_RBC'] = df['WBC_Count'] / df['RBC_Count']
df['WBC_Platelet'] = df['WBC_Count'] / df['Platelet_Count']
df['RBC_Hemoglobin'] = df['RBC_Count'] / df['Hemoglobin_Level']

print(df.head(5))
# Define features and target variable
X = df.drop(columns=["Patient_ID", "Leukemia_Status"])
y = df["Leukemia_Status"]

# Check for class imbalance
class_counts = y.value_counts()
print(f"Class counts:\n{class_counts}")

# Apply SMOTE to balance classes
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Check the new class distribution
class_counts_resampled = y_resampled.value_counts()
print(f"Class counts after SMOTE:\n{class_counts_resampled}")

# Save the processed dataset
df_resampled = pd.DataFrame(X_resampled, columns=X.columns)


# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

print(f"Training set size: {X_train.shape[0]} rows, {X_train.shape[1]} columns")
print(f"Test set size: {X_test.shape[0]} rows, {X_test.shape[1]} columns")


# MinMax scaling
scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Feature selection using SelectKBest
k_best = SelectKBest(score_func=chi2, k=10)
X_train_selected = k_best.fit_transform(X_train_scaled, y_train)
X_test_selected = k_best.transform(X_test_scaled)

# Get the selected feature names
selected_features = k_best.get_feature_names_out(input_features=X_train.columns)
print(f"Selected features: {selected_features}")


# Train the model
xgb_model = XGBClassifier(n_estimators=200, learning_rate=0.01, max_depth=5, random_state=42)

xgb_model.fit(X_train_selected, y_train)

# Make predictions
y_pred = xgb_model.predict(X_test_selected)
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred))


# Feature importance
plt.figure(figsize=(10, 6))
plt.barh(range(len(selected_features)), xgb_model.feature_importances_)
plt.yticks(range(len(selected_features)), selected_features)
plt.xlabel("Feature Importance")
plt.title("Feature Importance")
plt.show()


# Grid Search for Hyperparameter Tuning
param_grid = {
    'n_estimators': [200],
    'max_depth': [3],
    'learning_rate': [0.01],
    'subsample': [0.8],
}


# Perform Grid Search
grid_search = GridSearchCV(XGBClassifier(random_state=42), param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=1)

grid_search.fit(X_train_selected, y_train)
best_model = grid_search.best_estimator_
best_model.fit(X_train_selected, y_train)
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

# Make predictions with the best model
y_pred_best = best_model.predict(X_test_selected)

# Evaluate the best model
accuracy_best = accuracy_score(y_test, y_pred_best)
print(f"Accuracy of best model: {accuracy_best:.4f}")
print(classification_report(y_test, y_pred_best))


# Feature importance for the best model
plt.figure(figsize=(10, 6))
plt.barh(range(len(selected_features)), best_model.feature_importances_)
plt.yticks(range(len(selected_features)), selected_features)
plt.xlabel("Feature Importance")
plt.title("Feature Importance (Best Model)")
plt.show()


# Recursive Feature Elimination (RFECV)
xgb_model = XGBClassifier(n_estimators=500, random_state=42)
rfecv = RFECV(estimator=xgb_model, step=1, cv=3, min_features_to_select=10)
X_train_rfecv = rfecv.fit_transform(X_train_scaled, y_train)
X_test_rfecv = rfecv.transform(X_test_scaled)

xgb_model.fit(X_train_rfecv, y_train)
y_pred_rfecv = xgb_model.predict(X_test_rfecv)

accuracy_rfecv = accuracy_score(y_test, y_pred_rfecv)
print(f"Accuracy after RFECV: {accuracy_rfecv:.4f}")
print(classification_report(y_test, y_pred_rfecv))

# Ensure that selected features match the RFECV result
selected_features = X_train.columns[rfecv.support_]

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.barh(range(len(selected_features)), xgb_model.feature_importances_[:len(selected_features)], align='center')
plt.yticks(range(len(selected_features)), selected_features)
plt.xlabel('Feature Importance')
plt.title('Feature Importance Based on XGBoost Model')
plt.show()


# Train Logistic Regression
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train_selected, y_train)

# Predict on test data
y_pred_log_reg = log_reg.predict(X_test_selected)

# Evaluate
accuracy_log_reg = accuracy_score(y_test, y_pred_log_reg)
print(f"Logistic Regression Accuracy: {accuracy_log_reg:.4f}")
print(classification_report(y_test, y_pred_log_reg))

# Cat Boost Classifier , but not yet optimized through Gridsearch
cat_model = CatBoostClassifier(iterations=500, learning_rate=0.01, depth=5, random_state=42, verbose=0)

rfecv = RFECV(estimator=cat_model, step=1, cv=3, min_features_to_select=10)

X_train_rfecv = rfecv.fit_transform(X_train_scaled, y_train)
X_test_rfecv = rfecv.transform(X_test_scaled)

cat_model.fit(X_train_rfecv, y_train)
y_pred_rfecv = cat_model.predict(X_test_rfecv)
accuracy_rfecv = accuracy_score(y_test, y_pred_rfecv)
print(f"Accuracy after RFECV with CatBoost: {accuracy_rfecv:.4f}")
print(classification_report(y_test, y_pred_rfecv))

# Ensure that selected features match the RFECV result
selected_features = X_train.columns[rfecv.support_]

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.barh(range(len(selected_features)), cat_model.feature_importances_[:len(selected_features)], align='center')
plt.yticks(range(len(selected_features)), selected_features)
plt.xlabel('Feature Importance')
plt.title('Feature Importance Based on XGBoost Model')
plt.show()