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
plot_dir = "/Users/louisimhof/Desktop/University/Year 3/Research Project /Data/Healthcare/Plots/bestmodels/fairness"
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
X_test_s  = scaler.transform(X_test)


# Define hyperparameter grids
grid_lr  = {'C': [0.01]}
grid_xgb = {'n_estimators': [400], 'max_depth': [10], 'learning_rate': [0.1], 'subsample': [0.8]}
grid_cat = {'iterations': [1000], 'learning_rate': [0.05], 'depth': [10]}


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


# Summary of model performance
print("\nModel Summary:")

print(f"LR+RFECV:   {accuracy_score(y_test, y_lr_r):.4f}")
print(f"XGB+RFECV:  {accuracy_score(y_test, y_xgb_r):.4f}")



# Fairness-Checks für mehrere Modelle und sensible Gruppen
sensitive_cols = ["Gender", "Ethnicity", "Socioeconomic_Status", "Urban_Rural"]

# Modelle mit ihren jeweiligen Test-Features
models = {
    "LR+KBest":   (best_lr_r, X_test_r),
    "XGB+RFECV":  (best_xgb_r, X_test_r),

}

for col in sensitive_cols:
    print(f"\n=== Performance by {col} ===")
    for model_name, (model, X_feat) in models.items():
        print(f"\n-- {model_name} --")
        for grp in sorted(X_test[col].unique()):
            # Maske auf den originalen Test-Daten
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
    """
    Berechnet die Equalized Odds für jede Gruppe: TPR, FPR, FNR.
    """
    equalized_odds = {}

    for grp in groups:
        # Maske für die jeweilige Gruppe
        mask = (X_test[sensitive_col] == grp).values
        if mask.sum() == 0:
            continue

        y_true_grp = y_true[mask]
        y_pred_grp = y_pred[mask]

        # Berechnung der Confusion Matrix
        tn, fp, fn, tp = confusion_matrix(y_true_grp, y_pred_grp).ravel()

        # Berechnung von TPR, FPR, FNR
        tpr = tp / (tp + fn)  # True Positive Rate
        fpr = fp / (fp + tn)  # False Positive Rate
        fnr = fn / (tp + fn)  # False Negative Rate

        equalized_odds[grp] = {'TPR': tpr, 'FPR': fpr, 'FNR': fnr}

    return equalized_odds


def plot_disparity(equalized_odds, metric):
    """
    Visualisiert die Disparität eines bestimmten Metrics (TPR, FPR, FNR) zwischen den Gruppen.
    """
    groups = sorted(equalized_odds.keys())
    metric_values = [equalized_odds[group][metric] for group in groups]

    plt.figure(figsize=(8, 6))
    plt.bar(groups, metric_values, color='skyblue')
    plt.xlabel('Group')
    plt.ylabel(f'{metric}')
    plt.title(f'{metric} Disparity Across Groups for {model_name} by {sensitive_cols}')
    plt.xticks(rotation=45)

    # Speichern des Plots
    plot_filename = os.path.join(plot_dir, f"{model_name}_{sensitive_cols}_{metric}.png")
    plt.tight_layout()
    plt.savefig(plot_filename)
    plt.close()


# Berechnung von Equalized Odds für jedes Modell und jede Gruppe
sensitive_cols = ["Gender", "Ethnicity", "Socioeconomic_Status", "Urban_Rural"]
for col in sensitive_cols:
    print(f"\n=== Equalized Odds by {col} ===")
    for model_name, (model, X_feat) in models.items():
        print(f"\n-- {model_name} --")
        y_pred = model.predict(X_feat)

        equalized_odds = calculate_equalized_odds(y_test, y_pred, col, sorted(X_test[col].unique()))

        for grp, odds in equalized_odds.items():
            print(f"Group {grp}: TPR={odds['TPR']:.3f}, FPR={odds['FPR']:.3f}, FNR={odds['FNR']:.3f}")

        # Optional: Visualisierung der Ergebnisse (Disparität der Equalized Odds)
        plot_disparity(equalized_odds, 'TPR')
        plot_disparity(equalized_odds, 'FPR')
        plot_disparity(equalized_odds, 'FNR')


from fairlearn.metrics import MetricFrame
from sklearn.metrics import accuracy_score, classification_report, f1_score

# Berechne die Vorhersagen für das Modell
y_pred_lr = best_lr_r.predict(X_test_r)
y_pred_xgb = best_xgb_r.predict(X_test_r)

# Erstelle MetricFrames für beide Modelle mit mehreren sensitiven Merkmalen
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

# Visualisiere die Fairness-Metriken
metrics_lr_by_sensitive = metrics_lr.by_group
metrics_xgb_by_sensitive = metrics_xgb.by_group

# Plot for Logistic Regression Model
metrics_lr_by_sensitive[['accuracy', 'f1_score']].plot(kind='bar', figsize=(10, 6))
plt.title('Fairness Metrics for Logistic Regression')
plt.xlabel('Sensitive Groups')
plt.ylabel('Metrics')
plt.xticks(rotation=45)
plt.legend(title='Metric')
plt.tight_layout()
# Speichern des Plots
plt.savefig(os.path.join(plot_dir, "fairness_lr.png"))
plt.close()

# Plot for XGBoost Model
metrics_xgb_by_sensitive[['accuracy', 'f1_score']].plot(kind='bar', figsize=(10, 6))
plt.title('Fairness Metrics for XGBoost')
plt.xlabel('Sensitive Groups')
plt.ylabel('Metrics')
plt.xticks(rotation=45)
plt.legend(title='Metric')
plt.tight_layout()
# Speichern des Plots
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

    # Give extra room at the bottom for long labels
    plt.subplots_adjust(bottom=0.3, top=0.9)

    # Save with tight bounding box
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
report_disadvantaged_groups(metrics_xgb.by_group, "XGBoost (RFECV)",        metric='f1_score')
report_disadvantaged_groups(metrics_lr.by_group, "Logistic Regression (RFECV)", metric='accuracy')
report_disadvantaged_groups(metrics_xgb.by_group, "XGBoost (RFECV)",        metric='accuracy')


# Anteil jeder Gruppe in sensitiven Variablen anzeigen
for col in sensitive_cols:
    print(f"\nVerteilung in '{col}' (Testset):")
    value_counts = X_test[col].value_counts(normalize=True) * 100
    for val, pct in value_counts.items():
        print(f"  Gruppe {val}: {pct:.2f}%")
