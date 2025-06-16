import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import SelectKBest, chi2, RFECV
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib
matplotlib.use('Agg')  # Non-blocking backend for matplotlib
import matplotlib.pyplot as plt

# Set output directory for plots
plot_dir = '/Users/louisimhof/Desktop/University/Year 3/Research Project /Data/Healthcare/Plots'
os.makedirs(plot_dir, exist_ok=True)

# Load dataset
df = pd.read_csv('/Users/louisimhof/Desktop/University/Year 3/Research Project /Data/Healthcare/biased_leukemia_dataset.csv')
print(df.head())
print(df.isnull().sum())

# Outlier removal using IQR
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[column] >= lower) & (df[column] <= upper)]

for col in df.select_dtypes(include=[np.number]).columns:
    df = remove_outliers(df, col)

columns_to_check = ["WBC_Count", "RBC_Count", "Platelet_Count", "BMI", "Hemoglobin_Level", "Bone_Marrow_Blasts"]
for col in columns_to_check:
    df = remove_outliers(df, col)
print(f"Data after outlier removal: {df.shape} rows, {df.shape[1]} columns")

# Encode categorical columns
le = LabelEncoder()
for col in df.select_dtypes(include=["object"]).columns:
    df[col] = le.fit_transform(df[col])
    print(f"Encoded column: {col}")

# Duplicates check
print(f"Number of duplicate rows: {df.duplicated().sum()}")
# Feature Selection with Correlation Anaylsis
def select_features(df, threshold=0.7):
    base_features = ["WBC_Count", "RBC_Count", "Platelet_Count",
                     "Hemoglobin_Level", "Bone_Marrow_Blasts", "BMI"]

    df['ImmuneScore'] = df['Chronic_Illness'] + df['Immune_Disorders']
    df['RiskScore'] = (df['Smoking_Status'] + df['Alcohol_Consumption'] +
                       df['Radiation_Exposure'] + df['Infection_History'] +
                       df['Genetic_Mutation'] + df['Family_History'])

    df['log_WBC'] = np.log1p(df['WBC_Count'])
    df['log_Blasts'] = np.log1p(df['Bone_Marrow_Blasts'])

    df['Hemoglobin_WBC'] = df['Hemoglobin_Level'] / df['WBC_Count']
    df['Platelet_RBC'] = df['Platelet_Count'] / df['RBC_Count']

    corr_matrix = df.corr().abs()

    # remove correlated features
    features_to_drop = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if corr_matrix.iloc[i, j] >= threshold:
                colname = corr_matrix.columns[i]
                if colname not in base_features and colname not in features_to_drop:
                    features_to_drop.append(colname)

    df = df.drop(columns=features_to_drop)

    print(f"Removed Features because of correlation {features_to_drop}")
    return df


df = select_features(df)

print(df.head())

X = df.drop(columns=["Patient_ID", "Leukemia_Status"])
y = df["Leukemia_Status"]

# Apply SMOTE to handle class imbalance
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Scale the data using MinMaxScaler
scaler = MinMaxScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# Feature selection using SelectKBest
k_best = SelectKBest(score_func=chi2, k=20)
X_train_k = k_best.fit_transform(X_train_s, y_train)
X_test_k  = k_best.transform(X_test_s)
feat_k = k_best.get_feature_names_out(input_features=X_train.columns)

# Define hyperparameter grids
grid_lr  = {'C': [0.01]}
grid_xgb = {'n_estimators': [400], 'max_depth': [7], 'learning_rate': [0.1], 'subsample': [0.8]}
grid_cat = {'iterations': [1000], 'learning_rate': [0.05], 'depth': [7]}

# Logistic Regression + KBest
gs_lr_k = GridSearchCV(LogisticRegression(max_iter=1000, random_state=42), grid_lr, cv=5, scoring='accuracy', n_jobs=-1)
gs_lr_k.fit(X_train_k, y_train)
best_lr_k = gs_lr_k.best_estimator_
y_lr_k = best_lr_k.predict(X_test_k)
print("LR+KBest:")
print(classification_report(y_test, y_lr_k))

# Plot feature importance for LR+KBest
try:
    plt.figure(figsize=(10,6))
    plt.barh(range(len(feat_k)), np.abs(best_lr_k.coef_[0]))
    plt.yticks(range(len(feat_k)), feat_k)
    plt.title("LR+KBest Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "lr_kbest.png"))
    plt.close()
except Exception as e:
    print("Plot failed:", e)

# XGBoost + KBest
gs_xgb_k = GridSearchCV(XGBClassifier(random_state=42), grid_xgb, cv=5, scoring='accuracy', n_jobs=-1)
gs_xgb_k.fit(X_train_k, y_train)
best_xgb_k = gs_xgb_k.best_estimator_
y_xgb_k = best_xgb_k.predict(X_test_k)
print("XGB+KBest:")
print(classification_report(y_test, y_xgb_k))

# Plot feature importance for XGBoost+KBest
try:
    plt.figure(figsize=(10,6))
    plt.barh(range(len(feat_k)), best_xgb_k.feature_importances_)
    plt.yticks(range(len(feat_k)), feat_k)
    plt.title("XGB+KBest Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "xgb_kbest.png"))
    plt.close()
except Exception as e:
    print("Plot failed:", e)

# CatBoost + KBest
gs_cat_k = GridSearchCV(CatBoostClassifier(random_state=42, verbose=0), grid_cat, cv=5, scoring='accuracy', n_jobs=-1)
gs_cat_k.fit(X_train_k, y_train)
best_cat_k = gs_cat_k.best_estimator_
y_cat_k = best_cat_k.predict(X_test_k)
print("CAT+KBest:")
print(classification_report(y_test, y_cat_k))

# Plot feature importance for CatBoost+KBest
try:
    plt.figure(figsize=(10,6))
    plt.barh(range(len(feat_k)), best_cat_k.feature_importances_)
    plt.yticks(range(len(feat_k)), feat_k)
    plt.title("CAT+KBest Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "cat_kbest.png"))
    plt.close()
except Exception as e:
    print("Plot failed:", e)

# RFECV using XGBoost
xgb_rf_est = XGBClassifier(random_state=42)
rfecv = RFECV(estimator=xgb_rf_est, step=1, cv=5, min_features_to_select=10)
X_train_r = rfecv.fit_transform(X_train_s, y_train)
X_test_r  = rfecv.transform(X_test_s)
feat_r = X_train.columns[rfecv.support_]

# Logistic Regression + RFECV
gs_lr_r = GridSearchCV(LogisticRegression(max_iter=1000, random_state=42), grid_lr, cv=5, scoring='accuracy', n_jobs=-1)
gs_lr_r.fit(X_train_r, y_train)
best_lr_r = gs_lr_r.best_estimator_
y_lr_r = best_lr_r.predict(X_test_r)
print("LR+RFECV:")
print(classification_report(y_test, y_lr_r))

# Plot feature importance for LR+RFECV
try:
    plt.figure(figsize=(10,6))
    plt.barh(range(len(feat_r)), np.abs(best_lr_r.coef_[0]))
    plt.yticks(range(len(feat_r)), feat_r)
    plt.title("LR+RFECV Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "lr_rfecv.png"))
    plt.close()
except Exception as e:
    print("Plot failed:", e)

# XGBoost + RFECV
gs_xgb_r = GridSearchCV(XGBClassifier(random_state=42), grid_xgb, cv=5, scoring='accuracy', n_jobs=-1)
gs_xgb_r.fit(X_train_r, y_train)
best_xgb_r = gs_xgb_r.best_estimator_
y_xgb_r = best_xgb_r.predict(X_test_r)
print("XGB+RFECV:")
print(classification_report(y_test, y_xgb_r))

# Plot feature importance for XGBoost+RFECV
try:
    plt.figure(figsize=(10,6))
    plt.barh(range(len(feat_r)), best_xgb_r.feature_importances_)
    plt.yticks(range(len(feat_r)), feat_r)
    plt.title("XGB+RFECV Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "xgb_rfecv.png"))
    plt.close()
except Exception as e:
    print("Plot failed:", e)

# CatBoost + RFECV
gs_cat_r = GridSearchCV(CatBoostClassifier(random_state=42, verbose=0), grid_cat, cv=5, scoring='accuracy', n_jobs=-1)
gs_cat_r.fit(X_train_r, y_train)
best_cat_r = gs_cat_r.best_estimator_
y_cat_r = best_cat_r.predict(X_test_r)
print("CAT+RFECV:")
print(classification_report(y_test, y_cat_r))

# Plot feature importance for CatBoost+RFECV
try:
    plt.figure(figsize=(10,6))
    plt.barh(range(len(feat_r)), best_cat_r.feature_importances_)
    plt.yticks(range(len(feat_r)), feat_r)
    plt.title("CAT+RFECV Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "cat_rfecv.png"))
    plt.close()
except Exception as e:
    print("Plot failed:", e)

# Summary of model performance
print("\nModel Summary:")
print(f"LR+KBest:   {accuracy_score(y_test, y_lr_k):.4f}")
print(f"XGB+KBest:  {accuracy_score(y_test, y_xgb_k):.4f}")
print(f"CAT+KBest:  {accuracy_score(y_test, y_cat_k):.4f}")
print(f"LR+RFECV:   {accuracy_score(y_test, y_lr_r):.4f}")
print(f"XGB+RFECV:  {accuracy_score(y_test, y_xgb_r):.4f}")
print(f"CAT+RFECV:  {accuracy_score(y_test, y_cat_r):.4f}")

from sklearn.model_selection import learning_curve

def plot_learning_curve(estimator, X, y, title, filename):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 5),
        scoring='accuracy'
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    plt.figure()
    plt.plot(train_sizes, train_scores_mean, label="Training score", marker='o')
    plt.plot(train_sizes, test_scores_mean, label="Cross-validation score", marker='x')
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, filename))
    plt.close()


plot_learning_curve(best_lr_k, X_train_k, y_train, "LR+KBest Learning Curve", "lr_kbest_lc.png")
print("LR+KBest Learning Curve plotted.")
plot_learning_curve(best_xgb_k, X_train_k, y_train, "XGB+KBest Learning Curve", "xgb_kbest_lc.png")
print("XGB+KBest Learning Curve plotted.")
plot_learning_curve(best_cat_k, X_train_k, y_train, "CAT+KBest Learning Curve", "cat_kbest_lc.png")
print("CAT+KBest Learning Curve plotted.")

plot_learning_curve(best_lr_r, X_train_r, y_train, "LR+RFECV Learning Curve", "lr_rfecv_lc.png")
print("LR+RFECV Learning Curve plotted.")
plot_learning_curve(best_xgb_r, X_train_r, y_train, "XGB+RFECV Learning Curve", "xgb_rfecv_lc.png")
print("XGB+RFECV Learning Curve plotted.")
plot_learning_curve(best_cat_r, X_train_r, y_train, "CAT+RFECV Learning Curve", "cat_rfecv_lc.png")
print("CAT+RFECV Learning Curve plotted.")

