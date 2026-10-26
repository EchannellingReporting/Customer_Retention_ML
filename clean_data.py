
from pathlib import Path
import pandas as pd

# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "appointments.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATE_COLUMN = "Trns_Date"
MOBILE_COLUMN = "Mob_Number"
FEE_COLUMN = "ECH_Fee"

# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 65)
print("CUSTOMER RETENTION ML - DATA CLEANING")
print("=" * 65)

if not DATA_FILE.exists():
    raise FileNotFoundError(f"Input file not found: {DATA_FILE}")

df = pd.read_csv(DATA_FILE, dtype=str, low_memory=False)
df.columns = df.columns.str.strip()

required_columns = [
    "Reference_Number",
    "Mob_Number",
    "Hospital",
    "Doctor",
    "Specialization",
    "Trns_Date",
    "ECH_Fee",
    "Channel",
    "Town",
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

print(f"\nOriginal appointment rows: {len(df):,}")

# ============================================================
# 3. CLEAN TEXT COLUMNS
# ============================================================

text_columns = [
    "Reference_Number",
    "Mob_Number",
    "Hospital",
    "Doctor",
    "Specialization",
    "Channel",
    "Town",
]

for column in text_columns:
    df[column] = df[column].astype("string").str.strip()

# Convert common missing-value strings to actual missing values.
missing_tokens = ["", "nan", "none", "null", "nat"]

for column in text_columns:
    df[column] = df[column].mask(
        df[column].str.lower().isin(missing_tokens),
        pd.NA
    )

# Preserve mobile numbers as text, not numbers.
# This avoids losing leading zeros.

# ============================================================
# 4. CLEAN APPOINTMENT DATES
# ============================================================

df[DATE_COLUMN] = pd.to_datetime(
    df[DATE_COLUMN].astype("string").str.strip(),
    format="mixed",
    dayfirst=False,
    errors="coerce",
)

invalid_date_count = int(df[DATE_COLUMN].isna().sum())

print("\n--- DATE VALIDATION ---")
print(f"Rows with missing/invalid dates: {invalid_date_count:,}")

# Save records with invalid dates separately for review.
invalid_dates = df.loc[df[DATE_COLUMN].isna()].copy()

if not invalid_dates.empty:
    invalid_dates.to_csv(
        OUTPUT_DIR / "invalid_date_records.csv",
        index=False
    )

# Exclude records with invalid dates from the cleaned dataset.
df = df.loc[df[DATE_COLUMN].notna()].copy()

print(f"Earliest date: {df[DATE_COLUMN].min()}")
print(f"Latest date:   {df[DATE_COLUMN].max()}")

# ============================================================
# 5. CLEAN APPOINTMENT FEES
# ============================================================

# Convert fee values to numeric.
# Non-numeric or blank fees become missing values.
df[FEE_COLUMN] = pd.to_numeric(
    df[FEE_COLUMN]
        .astype("string")
        .str.replace(",", "", regex=False)
        .str.strip(),
    errors="coerce",
)

invalid_fee_count = int(df[FEE_COLUMN].isna().sum())

print("\n--- FEE VALIDATION ---")
print(f"Missing/non-numeric fees: {invalid_fee_count:,}")

# Do not automatically replace missing fees with zero.
# A missing fee is not necessarily a zero-fee appointment.

# ============================================================
# 6. CHECK DUPLICATE RECORDS
# ============================================================

duplicate_count = int(df.duplicated().sum())

print("\n--- DUPLICATE CHECK ---")
print(f"Completely duplicated rows: {duplicate_count:,}")

# Remove only completely identical rows.
# Do not remove rows based only on customer mobile number.
df = df.drop_duplicates().copy()

# ============================================================
# 7. SEPARATE RECORDS WITHOUT MOBILE NUMBERS
# ============================================================

missing_mobile_mask = (
    df[MOBILE_COLUMN].isna()
    | df[MOBILE_COLUMN].astype("string").str.strip().eq("")
)

missing_mobile_records = df.loc[missing_mobile_mask].copy()

print("\n--- MOBILE NUMBER CHECK ---")
print(f"Rows without mobile numbers: {len(missing_mobile_records):,}")

# Save separately so these records can be reviewed.
if not missing_mobile_records.empty:
    missing_mobile_records.to_csv(
        OUTPUT_DIR / "records_without_mobile.csv",
        index=False
    )

# Customer-level grouping requires a usable customer identifier.
# Keep these rows in the cleaned appointment dataset, but exclude
# them from the customer-level summary.
identified_df = df.loc[~missing_mobile_mask].copy()

# ============================================================
# 8. SAVE CLEANED APPOINTMENT DATA
# ============================================================

cleaned_file = OUTPUT_DIR / "cleaned_appointments.csv"

df.to_csv(cleaned_file, index=False)

print("\n--- CLEANED APPOINTMENT DATA ---")
print(f"Rows retained: {len(df):,}")
print(f"Saved to: {cleaned_file}")

# ============================================================
# 9. CREATE CUSTOMER-LEVEL SUMMARY
# ============================================================

# This is a descriptive summary of the full available history.
# It is NOT yet the ML training dataset: using full-history
# features for prediction would leak future information.
# We will build cutoff-based features and labels separately.

customer_summary = (
    identified_df
    .groupby(MOBILE_COLUMN, dropna=True)
    .agg(
        Total_Appointments=("Reference_Number", "size"),
        First_Appointment=(DATE_COLUMN, "min"),
        Last_Appointment=(DATE_COLUMN, "max"),
        Unique_Hospitals=("Hospital", "nunique"),
        Unique_Doctors=("Doctor", "nunique"),
        Unique_Specializations=("Specialization", "nunique"),
        Unique_Channels=("Channel", "nunique"),
        Unique_Towns=("Town", "nunique"),
        Total_ECH_Fee=("ECH_Fee", "sum"),
        Average_ECH_Fee=("ECH_Fee", "mean"),
    )
    .reset_index()
)

customer_summary["Days_Between_First_And_Last"] = (
    customer_summary["Last_Appointment"]
    - customer_summary["First_Appointment"]
).dt.days

customer_summary["Total_ECH_Fee"] = (
    customer_summary["Total_ECH_Fee"].round(2)
)

customer_summary["Average_ECH_Fee"] = (
    customer_summary["Average_ECH_Fee"].round(2)
)

summary_file = OUTPUT_DIR / "customer_summary.csv"

customer_summary.to_csv(summary_file, index=False)

# ============================================================
# 10. FINAL REPORT
# ============================================================

print("\n" + "=" * 65)
print("DATA CLEANING COMPLETED")
print("=" * 65)

print(f"Original rows: {len(pd.read_csv(DATA_FILE, usecols=[DATE_COLUMN])):,}")
print(f"Cleaned appointment rows: {len(df):,}")
print(f"Rows with identifiable customers: {len(identified_df):,}")
print(f"Unique customers: {customer_summary.shape[0]:,}")
print(f"Invalid dates excluded: {invalid_date_count:,}")
print(f"Rows without mobile numbers: {len(missing_mobile_records):,}")

print("\n--- OUTPUT FILES ---")
print(f"1. {cleaned_file}")
print(f"2. {summary_file}")

if not missing_mobile_records.empty:
    print(f"3. {OUTPUT_DIR / 'records_without_mobile.csv'}")

if invalid_date_count > 0:
    print(f"4. {OUTPUT_DIR / 'invalid_date_records.csv'}")

print("\nNext step: verify the complete observation period")
print("before creating Lost/Recurring labels for ML.")
