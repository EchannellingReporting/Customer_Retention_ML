
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = BASE_DIR / "models"

TRAINING_FILE = OUTPUT_DIR / "training_snapshots.csv"
SCORING_FILE = (
    OUTPUT_DIR / "scoring_customers_as_of_2026-09-16.csv"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MOBILE_COLUMN = "Mob_Number"
DATE_COLUMN = "Snapshot_Date"
TARGET_COLUMN = "Target"

# Only historical numeric features are allowed.
# Never add Mob_Number or future outcome columns here.
FEATURE_COLUMNS = [
    "Total_Appointments",
    "Appointments_Last_30_Days",
    "Appointments_Previous_30_Days",
    "Appointments_Last_60_Days",
    "Appointments_Last_90_Days",
    "Days_Since_Last_Appointment",
    "Customer_History_Days",
    "Unique_Hospitals",
    "Unique_Doctors",
    "Unique_Specializations",
    "Unique_Channels",
    "Unique_Towns",
    "Total_ECH_Fee",
    "Average_ECH_Fee",
    "Fee_Records_Count",
]

RANDOM_STATE = 42

print("=" * 70)
print("CUSTOMER RETENTION ML - MODEL TRAINING")
print("=" * 70)


# ============================================================
# 2. CHECK INPUT FILES
# ============================================================

if not TRAINING_FILE.exists():
    raise FileNotFoundError(
        f"Training file not found:\n{TRAINING_FILE}\n"
        "Run create_training_data.py first."
    )

if not SCORING_FILE.exists():
    raise FileNotFoundError(
        f"Scoring file not found:\n{SCORING_FILE}\n"
        "Run create_training_data.py first."
    )


# ============================================================
# 3. LOAD TRAINING DATA
# ============================================================

data = pd.read_csv(
    TRAINING_FILE,
    dtype={MOBILE_COLUMN: "string"},
    low_memory=False,
)

data.columns = data.columns.str.strip()

required_columns = (
    FEATURE_COLUMNS
    + [MOBILE_COLUMN, DATE_COLUMN, TARGET_COLUMN]
)

missing_columns = [
    col for col in required_columns
    if col not in data.columns
]

if missing_columns:
    raise ValueError(
        f"Training data is missing columns: {missing_columns}"
    )

data[DATE_COLUMN] = pd.to_datetime(
    data[DATE_COLUMN],
    errors="coerce",
)

data = data.dropna(
    subset=[DATE_COLUMN, TARGET_COLUMN]
).copy()

data = data.loc[
    data[TARGET_COLUMN].isin(["Lost", "Recurring"])
].copy()

if data.empty:
    raise ValueError("No valid labelled training records found.")

for column in FEATURE_COLUMNS:
    data[column] = pd.to_numeric(
        data[column],
        errors="coerce",
    )

print("\n--- TRAINING DATA ---")
print(f"Training rows: {len(data):,}")
print(f"Unique customers: {data[MOBILE_COLUMN].nunique():,}")

print("\nTarget distribution:")
print(data[TARGET_COLUMN].value_counts().to_string())

if data[TARGET_COLUMN].nunique() < 2:
    raise ValueError(
        "Training data must contain both Lost and Recurring labels."
    )


# ============================================================
# 4. TIME-BASED TRAIN / TEST SPLIT
# ============================================================

# The latest snapshot is reserved for the holdout test.
# Earlier snapshots are used for training.
# This avoids randomly mixing future snapshots into training.

snapshot_dates = sorted(
    data[DATE_COLUMN].dropna().unique()
)

if len(snapshot_dates) < 2:
    raise ValueError(
        "At least two snapshot dates are required for "
        "a time-based train/test split."
    )

test_date = pd.Timestamp(snapshot_dates[-1])

train_data = data.loc[
    data[DATE_COLUMN] < test_date
].copy()

test_data = data.loc[
    data[DATE_COLUMN] == test_date
].copy()

if train_data.empty or test_data.empty:
    raise ValueError(
        "Time-based split produced an empty train or test set."
    )

if train_data[TARGET_COLUMN].nunique() < 2:
    raise ValueError(
        "Training period contains only one target class. "
        "More historical snapshots may be needed."
    )

X_train = train_data[FEATURE_COLUMNS]
y_train = train_data[TARGET_COLUMN]

X_test = test_data[FEATURE_COLUMNS]
y_test = test_data[TARGET_COLUMN]

print("\n--- TIME-BASED SPLIT ---")
print(f"Training snapshots before: {test_date.date()}")
print(f"Training rows: {len(train_data):,}")
print(f"Holdout snapshot: {test_date.date()}")
print(f"Holdout rows: {len(test_data):,}")

print("\nHoldout target distribution:")
print(y_test.value_counts().to_string())


# ============================================================
# 5. DEFINE MODELS
# ============================================================

models = {
    "Logistic Regression": Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
        (
            "scaler",
            StandardScaler(),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
        ),
    ]),

    "Random Forest": Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=12,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
    ]),
}


# ============================================================
# 6. TRAIN AND EVALUATE
# ============================================================

results = []
fitted_models = {}
holdout_predictions = {}

for model_name, model in models.items():

    print("\n" + "-" * 70)
    print(f"Training: {model_name}")
    print("-" * 70)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    classifier = model.named_steps["classifier"]
    classes = list(classifier.classes_)

    lost_index = classes.index("Lost")

    probabilities = model.predict_proba(X_test)
    lost_probabilities = probabilities[:, lost_index]

    precision = precision_score(
        y_test,
        predictions,
        pos_label="Lost",
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        pos_label="Lost",
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        pos_label="Lost",
        zero_division=0,
    )

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    if y_test.nunique() == 2:
        roc_auc = roc_auc_score(
            (y_test == "Lost").astype(int),
            lost_probabilities,
        )
    else:
        roc_auc = np.nan

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Lost_Precision": precision,
        "Lost_Recall": recall,
        "Lost_F1_Score": f1,
        "ROC_AUC": roc_auc,
        "Training_Rows": len(train_data),
        "Holdout_Rows": len(test_data),
        "Holdout_Snapshot": str(test_date.date()),
    })

    fitted_models[model_name] = model

    holdout_predictions[model_name] = pd.DataFrame({
        MOBILE_COLUMN: test_data[MOBILE_COLUMN].values,
        "Snapshot_Date": test_data[DATE_COLUMN].values,
        "Actual_Target": y_test.values,
        "Predicted_Target": predictions,
        "Lost_Probability": lost_probabilities,
    })

    print(f"Accuracy:       {accuracy:.4f}")
    print(f"Lost Precision: {precision:.4f}")
    print(f"Lost Recall:    {recall:.4f}")
    print(f"Lost F1-score:  {f1:.4f}")

    if not np.isnan(roc_auc):
        print(f"ROC-AUC:        {roc_auc:.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=["Recurring", "Lost"],
            zero_division=0,
        )
    )

    print("Confusion matrix (actual rows, predicted columns):")
    print(
        pd.DataFrame(
            confusion_matrix(
                y_test,
                predictions,
                labels=["Recurring", "Lost"],
            ),
            index=["Actual Recurring", "Actual Lost"],
            columns=["Predicted Recurring", "Predicted Lost"],
        ).to_string()
    )


# ============================================================
# 7. SAVE METRICS AND HOLDOUT PREDICTIONS
# ============================================================

metrics_df = pd.DataFrame(results)

metrics_file = OUTPUT_DIR / "model_metrics.csv"

metrics_df.to_csv(
    metrics_file,
    index=False,
)

for model_name, result_df in holdout_predictions.items():

    safe_name = (
        model_name.lower()
        .replace(" ", "_")
    )

    result_file = (
        OUTPUT_DIR / f"holdout_{safe_name}_predictions.csv"
    )

    result_df.to_csv(
        result_file,
        index=False,
    )


# ============================================================
# 8. SELECT BASELINE MODEL
# ============================================================

# Select the model with the highest Lost F1 on the latest
# temporal holdout. Treat this as an initial comparison,
# not a final unbiased estimate after model selection.

best_row = metrics_df.sort_values(
    by="Lost_F1_Score",
    ascending=False,
).iloc[0]

best_name = best_row["Model"]
best_model = fitted_models[best_name]

model_file = MODEL_DIR / "best_model.joblib"

joblib.dump(
    {
        "model": best_model,
        "feature_columns": FEATURE_COLUMNS,
        "target_classes": ["Lost", "Recurring"],
        "selected_model": best_name,
        "training_cutoff": str(test_date.date()),
        "data_end_date": "2026-09-16",
    },
    model_file,
)

print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(
    metrics_df.sort_values(
        by="Lost_F1_Score",
        ascending=False,
    ).to_string(index=False)
)

print(f"\nSelected baseline model: {best_name}")
print(f"Saved model: {model_file}")


# ============================================================
# 9. SCORE CUSTOMERS AS OF SEPTEMBER 16, 2026
# ============================================================

# This is a historical scoring exercise.
# It cannot establish customer risk as of October 2026,
# because the dataset contains no later appointments.

scoring_data = pd.read_csv(
    SCORING_FILE,
    dtype={MOBILE_COLUMN: "string"},
    low_memory=False,
)

scoring_data.columns = scoring_data.columns.str.strip()

missing_scoring_features = [
    col for col in FEATURE_COLUMNS
    if col not in scoring_data.columns
]

if missing_scoring_features:
    raise ValueError(
        "Scoring file is missing features: "
        f"{missing_scoring_features}"
    )

for column in FEATURE_COLUMNS:
    scoring_data[column] = pd.to_numeric(
        scoring_data[column],
        errors="coerce",
    )

X_scoring = scoring_data[FEATURE_COLUMNS]

scoring_classifier = best_model.named_steps["classifier"]
scoring_classes = list(scoring_classifier.classes_)

lost_index = scoring_classes.index("Lost")
recurring_index = scoring_classes.index("Recurring")

scoring_probabilities = best_model.predict_proba(X_scoring)

lost_probability = scoring_probabilities[:, lost_index]
recurring_probability = scoring_probabilities[:, recurring_index]

scored = scoring_data[[MOBILE_COLUMN, "Snapshot_Date"]].copy()

scored["Lost_Probability"] = lost_probability
scored["Recurring_Probability"] = recurring_probability

scored["Predicted_Status"] = np.where(
    lost_probability >= 0.5,
    "Lost",
    "Recurring",
)

scored["Risk_Level"] = np.select(
    [
        lost_probability >= 0.70,
        lost_probability >= 0.40,
    ],
    [
        "High",
        "Medium",
    ],
    default="Low",
)

scored = scored.sort_values(
    by="Lost_Probability",
    ascending=False,
).reset_index(drop=True)

prediction_file = (
    OUTPUT_DIR / "customer_risk_predictions_as_of_2026-09-16.csv"
)

scored.to_csv(
    prediction_file,
    index=False,
)

print("\n--- CUSTOMER SCORING SUMMARY ---")
print(f"Customers scored: {len(scored):,}")

print("\nPredicted status:")
print(scored["Predicted_Status"].value_counts().to_string())

print("\nRisk level:")
print(scored["Risk_Level"].value_counts().to_string())

print(f"\nSaved customer predictions to:\n{prediction_file}")
print(f"Saved metrics to:\n{metrics_file}")

print("\n" + "=" * 70)
print("MODEL TRAINING AND SCORING COMPLETED")
print("=" * 70)

print(
    "IMPORTANT: These scores describe predictions based on "
    "historical data through 2026-09-16. They are not "
    "verified current customer statuses."
)
