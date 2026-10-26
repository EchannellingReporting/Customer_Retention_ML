
from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================
# CUSTOMER RETENTION ML - TRAINING DATA PREPARATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

APPOINTMENTS_FILE = OUTPUT_DIR / "cleaned_appointments.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATE_COLUMN = "Trns_Date"
MOBILE_COLUMN = "Mob_Number"
FEE_COLUMN = "ECH_Fee"

OBSERVATION_DAYS = 90
DATA_END_DATE = pd.Timestamp("2026-09-16")

# All dates below have a complete 90-day future window
# within the available appointment data.
SNAPSHOT_DATES = [
    pd.Timestamp("2026-01-31"),
    pd.Timestamp("2026-02-28"),
    pd.Timestamp("2026-03-20"),
    pd.Timestamp("2026-06-18"),
]

# ============================================================
# 1. LOAD CLEANED DATA
# ============================================================

print("=" * 65)
print("CUSTOMER RETENTION ML - TRAINING DATA PREPARATION")
print("=" * 65)

if not APPOINTMENTS_FILE.exists():
    raise FileNotFoundError(
        f"Cleaned appointments file not found:\n"
        f"{APPOINTMENTS_FILE}\n"
        "Run clean_data.py first."
    )

df = pd.read_csv(
    APPOINTMENTS_FILE,
    dtype={MOBILE_COLUMN: "string"},
    low_memory=False
)

df.columns = df.columns.str.strip()

required_columns = [
    MOBILE_COLUMN,
    DATE_COLUMN,
    FEE_COLUMN,
    "Hospital",
    "Doctor",
    "Specialization",
    "Channel",
    "Town",
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Required columns missing: {missing_columns}"
    )

# Parse dates consistently with the inspected source format.
df[DATE_COLUMN] = pd.to_datetime(
    df[DATE_COLUMN],
    errors="coerce",
    format="mixed",
    dayfirst=False
)

# Keep mobile numbers as identifiers, never numeric features.
df[MOBILE_COLUMN] = (
    df[MOBILE_COLUMN]
    .astype("string")
    .str.strip()
)

df[MOBILE_COLUMN] = df[MOBILE_COLUMN].mask(
    df[MOBILE_COLUMN].str.lower().isin(
        ["", "nan", "none", "null", "nat"]
    ),
    pd.NA
)

df[FEE_COLUMN] = pd.to_numeric(
    df[FEE_COLUMN],
    errors="coerce"
)

# Exclude records without usable dates or customer identifiers.
df = df.loc[
    df[DATE_COLUMN].notna()
    & df[MOBILE_COLUMN].notna()
].copy()

df = df.loc[
    df[DATE_COLUMN] <= DATA_END_DATE
].copy()

df = df.sort_values(DATE_COLUMN).reset_index(drop=True)

if df.empty:
    raise ValueError("No valid appointment records remain.")

actual_max_date = df[DATE_COLUMN].max().normalize()

print(f"\nAppointment rows: {len(df):,}")
print(f"Unique customers: {df[MOBILE_COLUMN].nunique():,}")
print(f"First appointment date: {df[DATE_COLUMN].min()}")
print(f"Last appointment date: {actual_max_date}")

if actual_max_date < DATA_END_DATE:
    print(
        "\nWARNING: Actual data ends before the expected "
        "2026-09-16 date. Snapshot windows will be checked "
        "against the actual available data."
    )

# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================

def create_customer_features(history, cutoff):
    """
    Use only appointments on or before cutoff.
    Return one feature row per customer.
    """

    if history.empty:
        return pd.DataFrame()

    cutoff = pd.Timestamp(cutoff)

    data = history.copy()

    # History must never contain appointments after cutoff.
    data = data.loc[data[DATE_COLUMN] <= cutoff].copy()

    if data.empty:
        return pd.DataFrame()

    rows = []

    for mobile, customer in data.groupby(
        MOBILE_COLUMN,
        sort=False
    ):
        dates = customer[DATE_COLUMN]

        first_date = dates.min()
        last_date = dates.max()

        last_30_start = cutoff - pd.Timedelta(days=30)
        last_60_start = cutoff - pd.Timedelta(days=60)
        last_90_start = cutoff - pd.Timedelta(days=90)

        last_30 = customer.loc[
            customer[DATE_COLUMN] > last_30_start
        ]

        previous_30 = customer.loc[
            (customer[DATE_COLUMN] > last_60_start)
            & (customer[DATE_COLUMN] <= last_30_start)
        ]

        last_60 = customer.loc[
            customer[DATE_COLUMN] > last_60_start
        ]

        last_90 = customer.loc[
            customer[DATE_COLUMN] > last_90_start
        ]

        fees = customer[FEE_COLUMN]

        rows.append({
            MOBILE_COLUMN: mobile,

            "Snapshot_Date": cutoff,

            "Total_Appointments": len(customer),

            "Appointments_Last_30_Days": len(last_30),

            "Appointments_Previous_30_Days": len(previous_30),

            "Appointments_Last_60_Days": len(last_60),

            "Appointments_Last_90_Days": len(last_90),

            "Days_Since_Last_Appointment": (
                cutoff - last_date
            ).days,

            "Customer_History_Days": (
                cutoff - first_date
            ).days,

            "Unique_Hospitals": customer["Hospital"].nunique(),

            "Unique_Doctors": customer["Doctor"].nunique(),

            "Unique_Specializations": (
                customer["Specialization"].nunique()
            ),

            "Unique_Channels": customer["Channel"].nunique(),

            "Unique_Towns": customer["Town"].nunique(),

            "Total_ECH_Fee": fees.sum(min_count=1),

            "Average_ECH_Fee": fees.mean(),

            "Fee_Records_Count": int(fees.notna().sum()),
        })

    features = pd.DataFrame(rows)

    # Numeric feature columns only.
    # Mob_Number is retained solely as a customer identifier.
    return features


# ============================================================
# 3. CREATE LABELS USING THE FUTURE 90-DAY WINDOW
# ============================================================

training_parts = []

print("\n--- SNAPSHOT AND LABEL CHECK ---")

for cutoff in SNAPSHOT_DATES:

    window_end = cutoff + pd.Timedelta(
        days=OBSERVATION_DAYS
    )

    print(f"\nSnapshot date: {cutoff.date()}")
    print(f"Future window end: {window_end.date()}")

    # A complete future observation window is mandatory.
    if window_end > actual_max_date:
        print(
            "SKIPPED: Not enough future data to observe "
            "the full 90-day window."
        )
        continue

    # Features use historical records only.
    history = df.loc[
        df[DATE_COLUMN] <= cutoff
    ].copy()

    features = create_customer_features(
        history,
        cutoff
    )

    if features.empty:
        print("SKIPPED: No customers available.")
        continue

    # Future appointments are used only to construct the label.
    future = df.loc[
        (df[DATE_COLUMN] > cutoff)
        & (df[DATE_COLUMN] <= window_end)
    ].copy()

    future_counts = (
        future.groupby(MOBILE_COLUMN)
        .size()
        .rename("Future_Appointment_Count")
    )

    features = features.merge(
        future_counts,
        left_on=MOBILE_COLUMN,
        right_index=True,
        how="left"
    )

    features["Future_Appointment_Count"] = (
        features["Future_Appointment_Count"]
        .fillna(0)
        .astype(int)
    )

    features["Target_Window_End"] = window_end

    features["Target"] = np.where(
        features["Future_Appointment_Count"] > 0,
        "Recurring",
        "Lost"
    )

    training_parts.append(features)

    print(f"Customers labelled: {len(features):,}")
    print(
        "Recurring:",
        int((features["Target"] == "Recurring").sum())
    )
    print(
        "Lost:",
        int((features["Target"] == "Lost").sum())
    )

if not training_parts:
    raise ValueError(
        "No valid training snapshots were created. "
        "Check the actual maximum date and snapshot dates."
    )

training_data = pd.concat(
    training_parts,
    ignore_index=True
)

# ============================================================
# 4. SAVE TRAINING DATA
# ============================================================

training_file = OUTPUT_DIR / "training_snapshots.csv"

training_data.to_csv(
    training_file,
    index=False
)

# ============================================================
# 5. CREATE CUSTOMER SCORING DATA AS OF SEP 16
# ============================================================

# This dataset contains historical features only.
# No target labels are created because future appointments
# after Sep 16 are not present in this dataset.

scoring_cutoff = actual_max_date

scoring_history = df.loc[
    df[DATE_COLUMN] <= scoring_cutoff
].copy()

scoring_data = create_customer_features(
    scoring_history,
    scoring_cutoff
)

scoring_file = (
    OUTPUT_DIR / "scoring_customers_as_of_2026-09-16.csv"
)

scoring_data.to_csv(
    scoring_file,
    index=False
)

# ============================================================
# 6. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("TRAINING DATA PREPARATION COMPLETED")
print("=" * 65)

print(f"Training rows: {len(training_data):,}")
print(
    "Unique customers across snapshots:",
    training_data[MOBILE_COLUMN].nunique()
)

print("\nOverall label distribution:")
print(training_data["Target"].value_counts().to_string())

print("\nSnapshot-wise label distribution:")
print(
    pd.crosstab(
        training_data["Snapshot_Date"],
        training_data["Target"]
    ).to_string()
)

print("\nOutput files:")
print(f"1. {training_file}")
print(f"2. {scoring_file}")

print(
    "\nIMPORTANT: Do not use Mob_Number, "
    "Future_Appointment_Count, Target_Window_End, "
    "or Target as model input features."
)

print(
    "The June 18 snapshot is suitable for a 90-day "
    "outcome check through September 16."
)

print(
    "The September 16 scoring file is retrospective: "
    "it cannot establish current October customer risk "
    "without newer appointment data."
)
