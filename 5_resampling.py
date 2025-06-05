import os
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import RFECV
from sklearn.metrics import accuracy_score, classification_report
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

plot_dir = "/Users/louisimhof/Desktop/University/Year 3/Research Project /Data/Healthcare/Plots/bestmodels/fairness/resampled"
os.makedirs(plot_dir, exist_ok=True)

df = pd.read_csv(
    '/Users/louisimhof/Desktop/University/Year 3/Research Project /Data/Healthcare/biased_leukemia_dataset.csv'
)

# Outlier removal (IQR)
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[column] >= lower) & (df[column] <= upper)]

for col in df.select_dtypes(include=[np.number]).columns:
    df = remove_outliers(df, col)

for col in ["WBC_Count", "RBC_Count", "Platelet_Count", "BMI", "Hemoglobin_Level", "Bone_Marrow_Blasts"]:
    df = remove_outliers(df, col)

# Encode categorical columns
encoders = {}
for col in df.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Feature Selection with Correlation Analysis
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
    features_to_drop = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if corr_matrix.iloc[i, j] >= threshold:
                colname = corr_matrix.columns[i]
                if colname not in base_features and colname not in features_to_drop:
                    features_to_drop.append(colname)

    df = df.drop(columns=features_to_drop)
    return df

df = select_features(df)

X = df.drop(columns=["Patient_ID", "Leukemia_Status"])
y = df["Leukemia_Status"]

# Sensitive-group resampling
def resample_sensitive_groups(df, sensitive_cols, target_col):
    result_df = pd.DataFrame()
    unique_combinations = df.groupby(sensitive_cols).size().reset_index().drop(0, axis=1)
    for _, combination in unique_combinations.iterrows():
        mask = (df[sensitive_cols] == combination.values).all(axis=1)
        group_data = df[mask]
        pos = group_data[group_data[target_col] == 1]
        neg = group_data[group_data[target_col] == 0]
        if pos.empty or neg.empty:
            continue
        if len(pos) < len(neg):
            pos_res = pos.sample(n=len(neg), replace=True, random_state=42)
            group_res = pd.concat([pos_res, neg])
        else:
            neg_res = neg.sample(n=len(pos), replace=True, random_state=42)
            group_res = pd.concat([pos, neg_res])
        result_df = pd.concat([result_df, group_res])
    return result_df

sensitive_cols = ["Gender", "Ethnicity", "Socioeconomic_Status", "Urban_Rural"]
df_resampled = resample_sensitive_groups(df, sensitive_cols, "Leukemia_Status")

X_resampled = df_resampled.drop(columns=["Patient_ID", "Leukemia_Status"])
y_resampled = df_resampled["Leukemia_Status"]

# SMOTE on resampled data
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_resampled, y_resampled)

# Train-test split
X_train_res, X_test_res, y_train_res, y_test_res = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42
)

# Scale
scaler_res = MinMaxScaler()
X_train_res_s = scaler_res.fit_transform(X_train_res)
X_test_res_s = scaler_res.transform(X_test_res)

# RFECV feature selection
rfecv = RFECV(
    estimator=XGBClassifier(random_state=42),
    step=1,
    cv=5,
    min_features_to_select=21
)
X_train_res_r = rfecv.fit_transform(X_train_res_s, y_train_res)
X_test_res_r = rfecv.transform(X_test_res_s)
feat_res_r = X_train_res.columns[rfecv.support_]

# GridSearchCV for best hyperparameters
grid_xgb = {'n_estimators': [100, 200, 300], 'max_depth': [4, 6, 8], 'learning_rate': [0.1, 0.05], 'subsample': [0.8]}
gs_xgb_res = GridSearchCV(
    XGBClassifier(random_state=42),
    grid_xgb,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
gs_xgb_res.fit(X_train_res_r, y_train_res)
best_params = gs_xgb_res.best_params_

# Prepare DMatrix for xgboost.train
dtrain = xgb.DMatrix(X_train_res_r, label=y_train_res)
dtest = xgb.DMatrix(X_test_res_r,  label=y_test_res)

params = {
    'objective': 'binary:logistic',
    'max_depth': best_params['max_depth'],
    'eta': best_params['learning_rate'],
    'subsample': best_params['subsample'],
    'eval_metric': ['logloss', 'auc', 'error'],
    'seed': 42
}

evals = [(dtrain, 'train'), (dtest, 'eval')]
evals_result = {}  # will store the evaluation history

bst = xgb.train(
    params,
    dtrain,
    num_boost_round=best_params['n_estimators'],
    evals=evals,
    early_stopping_rounds=25,
    evals_result=evals_result,
    verbose_eval=False
)

# Predictions
y_xgb_res = (bst.predict(dtest) > 0.5).astype(int)

print("XGB+RFECV (After Resampling & Early Stopping):")
print(classification_report(y_test_res, y_xgb_res))

# Feature importance plot (by weight)
importance = bst.get_score(importance_type='weight')
imp_list = [importance.get(f"f{i}", 0) for i in range(len(feat_res_r))]

plt.figure(figsize=(10,6))
plt.barh(range(len(feat_res_r)), imp_list)
plt.yticks(range(len(feat_res_r)), feat_res_r)
plt.title("XGB+RFECV Importance (After Resampling & Early Stopping)")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "xgb_rfecv_resampled_importance_2.png"))
plt.close()

# Learning curve plot
plt.figure(figsize=(10,6))
plt.plot(evals_result['train']['logloss'], label='Train Log Loss', color='blue')
plt.plot(evals_result['eval']['logloss'],  label='Test  Log Loss', color='orange')
if 'auc' in evals_result['train']:
    plt.plot(evals_result['train']['auc'],  label='Train AUC', color='green')
    plt.plot(evals_result['eval']['auc'],   label='Test  AUC', color='red')
if 'error' in evals_result['train']:
    plt.plot(evals_result['train']['error'], label='Train Error', color='cyan')
    plt.plot(evals_result['eval']['error'],  label='Test  Error', color='magenta')
plt.ylabel("Metric Value")
plt.xlabel("Boosting Round")
plt.title("Learning Curve (XGBoost): Log Loss, AUC & Error")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "xgb_rfecv_learning_curve_2.png"))
plt.close()

print("Learning Curve saved in:", os.path.join(plot_dir, "xgb_rfecv_learning_curve_2.png"))

# Additional evaluation metrics to assess overfitting
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score

# Evaluate classification performance
print("\n--- Evaluation Metrics ---")
acc = accuracy_score(y_test_res, y_xgb_res)
prec = precision_score(y_test_res, y_xgb_res)
rec = recall_score(y_test_res, y_xgb_res)
f1 = f1_score(y_test_res, y_xgb_res)
auc = roc_auc_score(y_test_res, bst.predict(dtest))
cm = confusion_matrix(y_test_res, y_xgb_res)

# Print individual metrics
print(f"Accuracy       : {acc:.4f}")
print(f"Precision      : {prec:.4f}")
print(f"Recall         : {rec:.4f}")
print(f"F1-Score       : {f1:.4f}")
print(f"ROC AUC Score  : {auc:.4f}")
print("Confusion Matrix:")
print(cm)

# Print difference between train and test metrics to detect overfitting
train_preds = (bst.predict(dtrain) > 0.5).astype(int)
train_acc = accuracy_score(y_train_res, train_preds)
train_f1 = f1_score(y_train_res, train_preds)
train_auc = roc_auc_score(y_train_res, bst.predict(dtrain))

print("\n--- Overfitting Check ---")
print(f"Train Accuracy : {train_acc:.4f} | Test Accuracy : {acc:.4f} | Diff: {train_acc - acc:.4f}")
print(f"Train F1       : {train_f1:.4f} | Test F1       : {f1:.4f} | Diff: {train_f1 - f1:.4f}")
print(f"Train AUC      : {train_auc:.4f} | Test AUC      : {auc:.4f} | Diff: {train_auc - auc:.4f}")

# Optional: warn if the model may be overfitting
threshold = 0.05
if (train_acc - acc) > threshold or (train_f1 - f1) > threshold or (train_auc - auc) > threshold:
    print("\n⚠️  Warning: The model shows signs of overfitting. Train/Test metric gaps are significant.")
else:
    print("\n✅  The model generalizes well. No significant overfitting detected.")


# Evaluation
train_preds = (bst.predict(dtrain) > 0.5).astype(int)
test_preds = (bst.predict(dtest) > 0.5).astype(int)

train_acc = accuracy_score(y_train_res, train_preds)
test_acc = accuracy_score(y_test_res, test_preds)

train_prec = precision_score(y_train_res, train_preds)
test_prec = precision_score(y_test_res, test_preds)

train_rec = recall_score(y_train_res, train_preds)
test_rec = recall_score(y_test_res, test_preds)

train_f1 = f1_score(y_train_res, train_preds)
test_f1 = f1_score(y_test_res, test_preds)

train_auc = roc_auc_score(y_train_res, bst.predict(dtrain))
test_auc = roc_auc_score(y_test_res, bst.predict(dtest))

print("\n--- Evaluation Overview ---")
print(f"Train Accuracy : {train_acc:.4f} | Test Accuracy : {test_acc:.4f} | Δ: {train_acc - test_acc:.4f}")
print(f"Train Precision: {train_prec:.4f} | Test Precision: {test_prec:.4f} | Δ: {train_prec - test_prec:.4f}")
print(f"Train Recall   : {train_rec:.4f} | Test Recall   : {test_rec:.4f} | Δ: {train_rec - test_rec:.4f}")
print(f"Train F1-Score : {train_f1:.4f} | Test F1-Score : {test_f1:.4f} | Δ: {train_f1 - test_f1:.4f}")
print(f"Train AUC      : {train_auc:.4f} | Test AUC      : {test_auc:.4f} | Δ: {train_auc - test_auc:.4f}")

print("\nConfusion Matrix (Test):")
print(confusion_matrix(y_test_res, test_preds))

if (train_acc - test_acc) > 0.05 or (train_f1 - test_f1) > 0.05 or (train_auc - test_auc) > 0.05:
    print("\n⚠️  Hinweis: Modell zeigt mögliche Overfitting-Tendenzen.")
else:
    print("\n✅  Modell generalisiert gut – keine auffälligen Overfitting-Anzeichen.")
