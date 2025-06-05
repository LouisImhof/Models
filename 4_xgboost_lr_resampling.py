import os
import pandas as pd
import numpy as np
from lightgbm import early_stopping
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, learning_curve
from sklearn.feature_selection import SelectKBest, chi2, RFECV
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
import matplotlib

matplotlib.use('Agg')  # Non-blocking backend for matplotlib
import matplotlib.pyplot as plt

# Set output directory for plots
plot_dir = "/Users/louisimhof/Desktop/University/Year 3/Research Project /Data/Healthcare/Plots/bestmodels/fairness/resampled"
os.makedirs(plot_dir, exist_ok=True)

# Load dataset
df = pd.read_csv(
    '/Users/louisimhof/Desktop/University/Year 3/Research Project /Data/Healthcare/biased_leukemia_dataset.csv')
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

# Encode categorical columns und Speicher der Encoder
encoders = {}
for col in df.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le
    print(f"Encoded column: {col} (classes: {le.classes_.tolist()})")

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
X_test_s = scaler.transform(X_test)

# Define hyperparameter grids
grid_lr = {'C': [0.01]}
grid_xgb = {'n_estimators': [400], 'max_depth': [10], 'learning_rate': [0.1], 'subsample': [0.8]}
grid_cat = {'iterations': [1000], 'learning_rate': [0.05], 'depth': [10]}

# RFECV using XGBoost
xgb_rf_est = XGBClassifier(random_state=42)
rfecv = RFECV(estimator=xgb_rf_est, step=1, cv=5, min_features_to_select=10)
X_train_r = rfecv.fit_transform(X_train_s, y_train)
X_test_r = rfecv.transform(X_test_s)
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
    plt.figure(figsize=(10, 6))
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
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(feat_r)), best_xgb_r.feature_importances_)
    plt.yticks(range(len(feat_r)), feat_r)
    plt.title("XGB+RFECV Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "xgb_rfecv.png"))
    plt.close()
except Exception as e:
    print("Plot failed:", e)

# Summary of model performance
print("\nModel Summary:")

print(f"LR+RFECV:   {accuracy_score(y_test, y_lr_r):.4f}")
print(f"XGB+RFECV:  {accuracy_score(y_test, y_xgb_r):.4f}")

# Fairness Checks
sensitive_cols = ["Gender", "Ethnicity", "Socioeconomic_Status", "Urban_Rural"]

models = {
    "LR+KBest": (best_lr_r, X_test_r),
    "XGB+RFECV": (best_xgb_r, X_test_r),

}

for col in sensitive_cols:
    print(f"\n=== Performance by {col} ===")
    for model_name, (model, X_feat) in models.items():
        print(f"\n-- {model_name} --")
        for grp in sorted(X_test[col].unique()):
            mask = (X_test[col] == grp).values
            if mask.sum() == 0:
                continue

            y_true = y_test[mask]
            y_pred = model.predict(X_feat[mask])

            acc = accuracy_score(y_true, y_pred)
            report = classification_report(y_true, y_pred, output_dict=True)

            print(f"Group {grp}: Acc={acc:.3f}, "
                  f"Prec(1)={report['1']['precision']:.3f}, "
                  f"Rec(1)={report['1']['recall']:.3f}, "
                  f"F1(1)={report['1']['f1-score']:.3f}")

from sklearn.metrics import confusion_matrix


def calculate_equalized_odds(y_true, y_pred, sensitive_col, groups):
    equalized_odds = {}

    for grp in groups:
        mask = (X_test[sensitive_col] == grp).values
        if mask.sum() == 0:
            continue

        y_true_grp = y_true[mask]
        y_pred_grp = y_pred[mask]

        tn, fp, fn, tp = confusion_matrix(y_true_grp, y_pred_grp).ravel()

        tpr = tp / (tp + fn)  # True Positive Rate
        fpr = fp / (fp + tn)  # False Positive Rate
        fnr = fn / (tp + fn)  # False Negative Rate

        equalized_odds[grp] = {'TPR': tpr, 'FPR': fpr, 'FNR': fnr}

    return equalized_odds


def plot_disparity(equalized_odds, metric):
    groups = sorted(equalized_odds.keys())
    metric_values = [equalized_odds[group][metric] for group in groups]

    plt.figure(figsize=(8, 6))
    plt.bar(groups, metric_values, color='skyblue')
    plt.xlabel('Group')
    plt.ylabel(f'{metric}')
    plt.title(f'{metric} Disparity Across Groups for {model_name} by {sensitive_cols}')
    plt.xticks(rotation=45)

    plot_filename = os.path.join(plot_dir, f"{model_name}_{sensitive_cols}_{metric}.png")
    plt.tight_layout()
    plt.savefig(plot_filename)
    plt.close()


# Equalized Odds for Sensitive Groups
sensitive_cols = ["Gender", "Ethnicity", "Socioeconomic_Status", "Urban_Rural"]
for col in sensitive_cols:
    print(f"\n=== Equalized Odds by {col} ===")
    for model_name, (model, X_feat) in models.items():
        print(f"\n-- {model_name} --")
        y_pred = model.predict(X_feat)

        equalized_odds = calculate_equalized_odds(y_test, y_pred, col, sorted(X_test[col].unique()))

        for grp, odds in equalized_odds.items():
            print(f"Group {grp}: TPR={odds['TPR']:.3f}, FPR={odds['FPR']:.3f}, FNR={odds['FNR']:.3f}")

        plot_disparity(equalized_odds, 'TPR')
        plot_disparity(equalized_odds, 'FPR')
        plot_disparity(equalized_odds, 'FNR')

from fairlearn.metrics import MetricFrame
from sklearn.metrics import accuracy_score, classification_report, f1_score

y_pred_lr = best_lr_r.predict(X_test_r)
y_pred_xgb = best_xgb_r.predict(X_test_r)

metrics_lr = MetricFrame(
    metrics={'accuracy': accuracy_score, 'f1_score': f1_score},
    y_true=y_test,
    y_pred=y_pred_lr,
    sensitive_features=X_test[["Gender", "Ethnicity", "Socioeconomic_Status", "Urban_Rural"]]
)

metrics_xgb = MetricFrame(
    metrics={'accuracy': accuracy_score, 'f1_score': f1_score},
    y_true=y_test,
    y_pred=y_pred_xgb,
    sensitive_features=X_test[["Gender", "Ethnicity", "Socioeconomic_Status", "Urban_Rural"]]
)


metrics_lr_by_sensitive = metrics_lr.by_group
metrics_xgb_by_sensitive = metrics_xgb.by_group


metrics_lr_by_sensitive[['accuracy', 'f1_score']].plot(kind='bar', figsize=(10, 6))
plt.title('Fairness Metrics for Logistic Regression')
plt.xlabel('Sensitive Groups')
plt.ylabel('Metrics')
plt.xticks(rotation=45)
plt.legend(title='Metric')
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "fairness_lr.png"))
plt.close()


metrics_xgb_by_sensitive[['accuracy', 'f1_score']].plot(kind='bar', figsize=(10, 6))
plt.title('Fairness Metrics for XGBoost')
plt.xlabel('Sensitive Groups')
plt.ylabel('Metrics')
plt.xticks(rotation=45)
plt.legend(title='Metric')
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "fairness_xgb.png"))
plt.close()


def plot_disadvantaged(df_by_group, model_name, metric='f1_score'):
    gm = df_by_group[metric]
    overall = gm.mean()
    colors = ['red' if v < overall else 'steelblue' for v in gm]
    labels = [", ".join(f"{col}={grp[i]}" for i, col in enumerate(df_by_group.index.names))
              for grp in gm.index]

    plt.figure(figsize=(12, 4))
    plt.bar(labels, gm, color=colors)
    plt.axhline(overall, color='black', linestyle='--', label=f'Overall {metric}')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel(metric)
    plt.title(f"{model_name} — {metric} by group")
    plt.legend()
    plt.subplots_adjust(bottom=0.35)

    plt.subplots_adjust(bottom=0.3, top=0.9)

    fname = os.path.join(plot_dir, f"{model_name}_{metric}_disadvantaged.png")
    plt.savefig(fname, bbox_inches='tight')
    plt.close()


def report_disadvantaged_groups(df_by_group, model_name, metric='f1_score'):
    group_metrics = df_by_group[metric]
    overall = group_metrics.mean()
    disadvantaged = group_metrics[group_metrics < overall]
    if disadvantaged.empty:
        print(f"{model_name}: no group below average {metric}")
        return
    print(f"{model_name} – groups with {metric} below overall mean ({overall:.3f}):")
    for grp, val in disadvantaged.items():
        decoded = {col: encoders[col].inverse_transform([grp[i]])[0]
                   for i, col in enumerate(sensitive_cols)}
        decoded_str = ", ".join(f"{col}={decoded[col]}" for col in sensitive_cols)
        print(f"  [{decoded_str}]: {val:.3f}")
    print()


report_disadvantaged_groups(metrics_lr.by_group, "Logistic Regression (RFECV)", metric='f1_score')
report_disadvantaged_groups(metrics_xgb.by_group, "XGBoost (RFECV)", metric='f1_score')
report_disadvantaged_groups(metrics_lr.by_group, "Logistic Regression (RFECV)", metric='accuracy')
report_disadvantaged_groups(metrics_xgb.by_group, "XGBoost (RFECV)", metric='accuracy')


for col in sensitive_cols:
    print(f"\nDistribution in '{col}' (Testset):")
    value_counts = X_test[col].value_counts(normalize=True) * 100
    for val, pct in value_counts.items():
        print(f"  Gruppe {val}: {pct:.2f}%")

from sklearn.utils import resample


def resample_sensitive_groups(df, sensitive_cols, target_col):
    if df.empty:
        raise ValueError("Data is empty. Cannot resample.")

    result_df = pd.DataFrame()

    unique_combinations = df.groupby(sensitive_cols).size().reset_index().drop(0, axis=1)
    print(f"[Debug] Number of uniques combinations: {len(unique_combinations)}")

    for _, combination in unique_combinations.iterrows():

        group_filter = (df[sensitive_cols] == combination.values).all(axis=1)
        group_data = df[group_filter]

        print(f"[Debug] Group {combination.to_dict()} has {len(group_data)} cases.")

        pos_examples = group_data[group_data[target_col] == 1]
        neg_examples = group_data[group_data[target_col] == 0]

        if pos_examples.empty or neg_examples.empty:
            print(f"[Warning] Group {combination.to_dict()} has no cases. Skip.")
            continue

        if len(pos_examples) < len(neg_examples):
            pos_resampled = resample(pos_examples, replace=True, n_samples=len(neg_examples), random_state=42)
            group_resampled = pd.concat([pos_resampled, neg_examples])
        else:
            neg_resampled = resample(neg_examples, replace=True, n_samples=len(pos_examples), random_state=42)
            group_resampled = pd.concat([pos_examples, neg_resampled])

        result_df = pd.concat([result_df, group_resampled])

    print(f"[Debug] After Resampling: {result_df.shape[0]} Zeilen.")
    return result_df


for col in sensitive_cols:
    print(f"\nDistribution in '{col}' (Testset):")
    value_counts = X_test[col].value_counts(normalize=True) * 100
    for val, pct in value_counts.items():
        print(f"  Group {val}: {pct:.2f}%")

sensitive_cols = ["Gender", "Ethnicity", "Socioeconomic_Status", "Urban_Rural"]
target_col = "Leukemia_Status"
df_resampled = resample_sensitive_groups(df, sensitive_cols, target_col)
print(f"Data after sensitive group resampling: {df_resampled.shape} rows.")

X_resampled = df_resampled.drop(columns=["Patient_ID", "Leukemia_Status"])
y_resampled = df_resampled["Leukemia_Status"]

# Apply SMOTE for further class balancing
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_resampled, y_resampled)

# Train-Test-Split
X_train_res, X_test_res, y_train_res, y_test_res = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Feature Scaling
scaler_res = MinMaxScaler()
X_train_res_s = scaler_res.fit_transform(X_train_res)
X_test_res_s = scaler_res.transform(X_test_res)

# Model Training with Resampled Data

# RFECV with Logistic Regression
rfecv = RFECV(estimator=XGBClassifier(random_state=42), step=1, cv=5, min_features_to_select=20)
X_train_res_r = rfecv.fit_transform(X_train_res_s, y_train_res)
X_test_res_r = rfecv.transform(X_test_res_s)
feat_res_r = X_train_res.columns[rfecv.support_]

gs_lr_res = GridSearchCV(LogisticRegression(max_iter=1000, random_state=42), grid_lr, cv=5, scoring='accuracy',
                         n_jobs=-1)
gs_lr_res.fit(X_train_res_r, y_train_res)
best_lr_res = gs_lr_res.best_estimator_
y_lr_res = best_lr_res.predict(X_test_res_r)

# Logistic Regression Ergebnis
print("LR+RFECV (After Resampling):")
print(classification_report(y_test_res, y_lr_res))

# Plot Feature Importance for Logistic Regression
try:
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(feat_res_r)), np.abs(best_lr_res.coef_[0]))
    plt.yticks(range(len(feat_res_r)), feat_res_r)
    plt.title("LR+RFECV Importance (After Resampling)")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "lr_rfecv_resampled.png"))
    plt.close()
except Exception as e:
    print("Plot failed:", e)

# XGBoost Model Training
gs_xgb_res = GridSearchCV(XGBClassifier(random_state=42), grid_xgb, cv=5, scoring='accuracy', n_jobs=-1)
gs_xgb_res.fit(X_train_res_r, y_train_res)
best_xgb_res = gs_xgb_res.best_estimator_
y_xgb_res = best_xgb_res.predict(X_test_res_r)

# XGBoost Ergebnis
print("XGB+RFECV (After Resampling):")
print(classification_report(y_test_res, y_xgb_res))

# Plot Feature Importance for XGBoost
try:
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(feat_res_r)), best_xgb_res.feature_importances_)
    plt.yticks(range(len(feat_res_r)), feat_res_r)
    plt.title("XGB+RFECV Importance (After Resampling)")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "xgb_rfecv_resampled.png"))
    plt.close()
except Exception as e:
    print("Plot failed:", e)


models_resampled = {
    "LR+KBest (Resampled)": (best_lr_res, X_test_res_r),
    "XGB+RFECV (Resampled)": (best_xgb_res, X_test_res_r)
}

for col in sensitive_cols:
    print(f"\n=== Performance by {col} (After Resampling) ===")
    for model_name, (model, X_feat) in models_resampled.items():
        print(f"\n-- {model_name} --")
        for grp in sorted(X_test_res[col].unique()):
            mask = (X_test_res[col] == grp).values
            if mask.sum() == 0:
                continue
            y_true = y_test_res[mask]
            y_pred = model.predict(X_feat[mask])

            acc = accuracy_score(y_true, y_pred)
            report = classification_report(y_true, y_pred, output_dict=True)

            print(f"Group {grp}: Acc={acc:.3f}, "
                  f"Prec(1)={report['1']['precision']:.3f}, "
                  f"Rec(1)={report['1']['recall']:.3f}, "
                  f"F1(1)={report['1']['f1-score']:.3f}")


y_train_pred = best_xgb_res.predict(X_train_res_r)
y_test_pred = best_xgb_res.predict(X_test_res_r)


print("Train Data:")
print(f"Accuracy: {accuracy_score(y_train_res, y_train_pred):.4f}")
print(f"F1-Score: {f1_score(y_train_res, y_train_pred):.4f}")


print("\nTest Data:")
print(f"Accuracy: {accuracy_score(y_test_res, y_test_pred):.4f}")
print(f"F1-Score: {f1_score(y_test_res, y_test_pred):.4f}")


best_xgb_res_lc = XGBClassifier(**gs_xgb_res.best_params_, random_state=4, eval_metric=["logloss", "auc", "error"])


eval_set = [(X_train_res_r, y_train_res), (X_test_res_r, y_test_res)]

best_xgb_res_lc.fit(
    X_train_res_r,
    y_train_res,
    eval_set=eval_set,
    verbose=0
)

results = best_xgb_res_lc.evals_result()
plt.figure(figsize=(10, 6))

# Log Loss Plot
plt.plot(results['validation_0']['logloss'], label='Train Log Loss', color='blue')
plt.plot(results['validation_1']['logloss'], label='Test Log Loss', color='orange')

# AUC Plot
if "auc" in results['validation_0']:
    plt.plot(results['validation_0']['auc'], label='Train AUC', color='green')
    plt.plot(results['validation_1']['auc'], label='Test AUC', color='red')

# Error Plot
if "error" in results['validation_0']:
    plt.plot(results['validation_0']['error'], label='Train Error', color='cyan')
    plt.plot(results['validation_1']['error'], label='Test Error', color='magenta')

plt.ylabel("Metric Value")
plt.xlabel("Iterations")
plt.title("Learning Curve (XGBoost): Log Loss, AUC & Error")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "xgb_rfecv_learning_curve_full.png"))
plt.close()

print("Learning Curve saved in:", os.path.join(plot_dir, "xgb_rfecv_learning_curve_full.png"))

from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference, MetricFrame
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


def calculate_fairness_metrics(y_true, y_pred, sensitive_features, scenario):
    print(f"=== Fairness Metrics [{scenario}] ===")

    metric_frame = MetricFrame(
        metrics={
            'Accuracy': accuracy_score,
            'Recall': recall_score,
            'Precision': precision_score,
            'F1-Score': f1_score
        },
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features
    )

    print("\nFairness Metrics per Group:")
    print(metric_frame.by_group)

    eod = equalized_odds_difference(
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features
    )
    print(f"\nEqualized Odds Difference: {eod:.3f}")

    dpd = demographic_parity_difference(
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features
    )
    print(f"Demographic Parity Difference: {dpd:.3f}\n")

    return metric_frame


print("Before Resampling:")
sensitive_features_pre = X_test[["Gender", "Ethnicity", "Socioeconomic_Status", "Urban_Rural"]]
fairness_metrics_pre = calculate_fairness_metrics(
    y_true=y_test,
    y_pred=y_xgb_r,
    sensitive_features=sensitive_features_pre,
    scenario="Before Resampling"
)

print("After Resampling:")
sensitive_features_post = X_test_res[["Gender", "Ethnicity", "Socioeconomic_Status", "Urban_Rural"]]
fairness_metrics_post = calculate_fairness_metrics(
    y_true=y_test_res,
    y_pred=y_xgb_res,
    sensitive_features=sensitive_features_post,
    scenario="After Resampling"
)


def plot_fairness_comparison(metric_frame_pre, metric_frame_post, metric_name, save_path=None):
    groups = [
        ", ".join(f"{level}" for level in grp) for grp in metric_frame_pre.by_group.index
    ]
    values_pre = metric_frame_pre.by_group[metric_name].values
    values_post = metric_frame_post.by_group[metric_name].values

    x = range(len(groups))

    plt.figure(figsize=(14, 7))

    plt.bar(x, values_pre, alpha=0.6, label=f"Before Resampling ({metric_name})", color='blue', width=0.4)

    plt.bar([pos + 0.4 for pos in x], values_post, alpha=0.6,
            label=f"After Resampling ({metric_name})", color='orange', width=0.4)

    plt.xticks([pos + 0.2 for pos in x], groups, rotation=90)
    plt.xlabel("Groups")
    plt.ylabel(metric_name)
    plt.title(f"Fairness Comparison ({metric_name}) for Groups")
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.close()


plot_fairness_comparison(
    metric_frame_pre=fairness_metrics_pre,
    metric_frame_post=fairness_metrics_post,
    metric_name="Recall",
    save_path=os.path.join(plot_dir, "fairness_equalized_odds_comparison.png")
)

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


plot_learning_curve(best_lr_res, X_train_res_r, y_train_res, "LR+RFECV Learning Curve Resampled", "lr_rfecv_lc_resampled.png")
print("Learning Curve for LR+RFECV saved in:", os.path.join(plot_dir, "lr_rfecv_lc_resampled.png"))
plot_learning_curve(best_xgb_res_lc, X_train_res_r, y_train_res, "XGB+RFECV Learning Curve Resampled", "xgb_rfecv_lc_resampled.png")
print("Learning Curve for XGB+RFECV saved in:", os.path.join(plot_dir, "xgb_rfecv_lc_resampled.png"))

