
from pathlib import Path
import pandas as pd

# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "appointments.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

DATE_COLUMN = "Trns_Date"
EXPECTED_CUTOFF = pd.Timestamp("2026-09-30")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 65)
print("CUSTOMER RETENTION ML - DATA INSPECTION")
print("=" * 65)

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"CSV file not found:\n{DATA_FILE}\n"
        "Check the data folder and file name."
    )

df = pd.read_csv(
    DATA_FILE,
    dtype=str,
    low_memory=False
)

df.columns = df.columns.str.strip()

print("\n--- DATASET OVERVIEW ---")
print(f"File: {DATA_FILE}")
print(f"Total rows: {len(df):,}")
print(f"Total columns: {len(df.columns):,}")

print("\n--- COLUMN NAMES ---")
for column in df.columns:
    print(f"- {column}")

if DATE_COLUMN not in df.columns:
    raise KeyError(
        f"Date column '{DATE_COLUMN}' not found.\n"
        f"Available columns: {df.columns.tolist()}"
    )


# ============================================================
# 3. MISSING AND BLANK VALUES
# ============================================================

print("\n--- MISSING VALUES ---")

missing_counts = df.isna().sum()
found_missing = False

for column, count in missing_counts.items():
    if count > 0:
        print(f"{column}: {count:,}")
        found_missing = True

if not found_missing:
    print("No missing values detected.")

print("\n--- BLANK VALUES ---")

found_blanks = False

for column in df.columns:
    blank_count = (
        df[column]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )

    if blank_count > 0:
        print(f"{column}: {blank_count:,}")
        found_blanks = True

if not found_blanks:
    print("No blank values detected.")


# ============================================================
# 4. INSPECT ORIGINAL DATE VALUES
# ============================================================

print("\n" + "=" * 65)
print("DATE COLUMN DIAGNOSTICS")
print("=" * 65)

raw_dates = df[DATE_COLUMN].astype("string").str.strip()

missing_date_mask = (
    raw_dates.isna()
    | raw_dates.str.lower().isin(
        ["", "nan", "none", "null", "nat"]
    )
)

nonblank_dates = raw_dates[~missing_date_mask]

print(f"\nDate column: {DATE_COLUMN}")
print(f"Missing/blank dates: {missing_date_mask.sum():,}")

print("\n--- SAMPLE RAW DATE VALUES ---")

for value in nonblank_dates.drop_duplicates().head(20):
    print(repr(value))


# ============================================================
# 5. PARSE DATES AS MM/DD/YYYY
# ============================================================

# Example:
# 1/2/2026  = January 2, 2026
# 1/13/2026 = January 13, 2026
#
# Based on the raw samples, use month-first parsing.
# Invalid values are converted to NaT so they can be inspected.

parsed_dates = pd.to_datetime(
    raw_dates,
    errors="coerce",
    format="mixed",
    dayfirst=False
)

df[DATE_COLUMN] = parsed_dates

invalid_date_mask = (
    ~missing_date_mask
    & df[DATE_COLUMN].isna()
)

print("\n--- DATE PARSING RESULTS ---")
print("Parsing format: MM/DD/YYYY (month-first)")
print(f"Total rows: {len(df):,}")
print(f"Successfully parsed: {df[DATE_COLUMN].notna().sum():,}")
print(f"Missing dates: {missing_date_mask.sum():,}")
print(f"Invalid nonblank dates: {invalid_date_mask.sum():,}")

valid_dates = df[DATE_COLUMN].dropna()

if not valid_dates.empty:
    print(f"Earliest appointment: {valid_dates.min()}")
    print(f"Latest appointment: {valid_dates.max()}")
else:
    print("No valid dates could be parsed.")


# ============================================================
# 6. CHECK RECORDS AFTER SEPTEMBER 30, 2026
# ============================================================

print("\n--- RECORDS AFTER 2026-09-30 ---")

after_cutoff = df.loc[
    df[DATE_COLUMN].notna()
    & (df[DATE_COLUMN] > EXPECTED_CUTOFF)
]

print(f"Rows after Sep 30: {len(after_cutoff):,}")

if not after_cutoff.empty:
    print("\nLatest dates after Sep 30:")

    print(
        after_cutoff[[DATE_COLUMN]]
        .sort_values(DATE_COLUMN, ascending=False)
        .head(10)
        .to_string(index=False)
    )
else:
    print("No records after September 30, 2026.")


# ============================================================
# 7. CHECK INVALID DATE EXAMPLES
# ============================================================

print("\n--- INVALID DATE EXAMPLES ---")

invalid_examples = (
    raw_dates[invalid_date_mask]
    .drop_duplicates()
    .head(20)
)

if invalid_examples.empty:
    print("No invalid nonblank date examples found.")
else:
    for value in invalid_examples:
        print(repr(value))


# ============================================================
# 8. DUPLICATE RECORD CHECK
# ============================================================

print("\n--- DUPLICATE RECORDS ---")

duplicate_count = int(df.duplicated().sum())

print(f"Completely duplicated rows: {duplicate_count:,}")


# ============================================================
# 9. SAVE DATE DIAGNOSTICS
# ============================================================

# Only date-related diagnostics are exported.
# Mobile numbers and other customer details are not included.

diagnostics = pd.DataFrame({
    "Original_Date_Value": raw_dates,
    "Parsed_Date": df[DATE_COLUMN],
    "Date_Is_Missing": missing_date_mask,
    "Date_Is_Invalid": invalid_date_mask,
    "Date_After_2026_09_30": (
        df[DATE_COLUMN].notna()
        & (df[DATE_COLUMN] > EXPECTED_CUTOFF)
    )
})

diagnostic_file = OUTPUT_DIR / "date_diagnostics.csv"

diagnostics.to_csv(
    diagnostic_file,
    index=False
)

print("\n--- OUTPUT FILE ---")
print(f"Saved diagnostics to:\n{diagnostic_file}")


# ============================================================
# 10. FINAL STATUS
# ============================================================

print("\n" + "=" * 65)
print("DATA INSPECTION COMPLETED")
print("=" * 65)

if not valid_dates.empty:
    if after_cutoff.empty:
        print(
            "Date check: No parsed appointments after "
            "September 30, 2026."
        )
    else:
        print(
            "WARNING: Records exist after September 30, 2026. "
            "Verify the original CSV and date format."
        )

if invalid_date_mask.any():
    print(
        "WARNING: Some nonblank dates could not be parsed. "
        "Inspect the invalid date examples."
    )

print(
    "\nDo not train the ML model until the date range "
    "has been verified."
)
