
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best_model.joblib"

MOBILE = "Mob_Number"
DATE = "Trns_Date"
FEE = "ECH_Fee"

CATEGORY_COLUMNS = [
    "Hospital",
    "Doctor",
    "Specialization",
    "Channel",
    "Town",
]

st.set_page_config(
    page_title="Customer Retention ML",
    page_icon="📊",
    layout="wide",
)

st.title("Customer Retention Prediction")
st.write(
    "Upload an appointment CSV to calculate customer features "
    "and predict Lost probability, predicted status and risk level."
)

if not MODEL_PATH.exists():
    st.error(f"Trained model not found: {MODEL_PATH}")
    st.stop()


@st.cache_resource
def load_artifact():
    return joblib.load(MODEL_PATH)


artifact = load_artifact()
model = artifact["model"]
feature_columns = artifact["feature_columns"]
training_cutoff = pd.Timestamp(
    artifact.get("data_end_date", "2026-09-16")
)

st.caption(
    f"Model training data cutoff: {training_cutoff.date()}"
)

st.info(
    "The app uses the latest valid appointment date in your uploaded "
    "file as the scoring date. Only customers with usable mobile "
    "numbers can receive individual predictions."
)

uploaded_file = st.file_uploader(
    "Upload raw appointment CSV",
    type=["csv"],
)

if uploaded_file is not None:
    try:
        # Read mobile numbers as text to preserve leading zeros.
        raw = pd.read_csv(
            uploaded_file,
            dtype={MOBILE: "string"},
            low_memory=False,
        )
        raw.columns = raw.columns.str.strip()

        required_columns = [
            MOBILE,
            DATE,
            FEE,
            *CATEGORY_COLUMNS,
        ]

        missing_columns = [
            col for col in required_columns
            if col not in raw.columns
        ]

        if missing_columns:
            st.error(
                "Required columns are missing: "
                + ", ".join(missing_columns)
            )
            st.write(
                "Expected columns include Mob_Number, Trns_Date, "
                "ECH_Fee, Hospital, Doctor, Specialization, Channel "
                "and Town."
            )
            st.stop()

        # Clean identifiers and parse dates.
        raw[MOBILE] = raw[MOBILE].str.strip()
        raw[MOBILE] = raw[MOBILE].replace("", pd.NA)

        # Uses pandas' default date interpretation.
        # Check the detected dates below before accepting the results.
        raw[DATE] = pd.to_datetime(
            raw[DATE],
            errors="coerce",
        )

        raw[FEE] = pd.to_numeric(
            raw[FEE],
            errors="coerce",
        )

        # Blank category values should not count as distinct categories.
        for col in CATEGORY_COLUMNS:
            raw[col] = raw[col].astype("string").str.strip()
            raw[col] = raw[col].replace("", pd.NA)

        invalid_dates = int(raw[DATE].isna().sum())
        missing_mobile = int(raw[MOBILE].isna().sum())

        dated_rows = raw.loc[raw[DATE].notna()].copy()

        if dated_rows.empty:
            st.error("No valid appointment dates were found.")
            st.stop()

        cutoff = dated_rows[DATE].max().normalize()

        st.subheader("1. Uploaded data overview")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Uploaded rows", f"{len(raw):,}")
        c2.metric(
            "Valid appointment dates",
            f"{raw[DATE].notna().sum():,}",
        )
        c3.metric("Rows without mobile", f"{missing_mobile:,}")
        c4.metric("Invalid or missing dates", f"{invalid_dates:,}")

        st.write(
            f"**Detected date range:** "
            f"{dated_rows[DATE].min().date()} to {cutoff.date()}"
        )

        st.warning(
            "Check that the detected date range matches your source "
            "CSV. If dates such as 1/2/2026 are interpreted incorrectly, "
            "the source date format must be specified explicitly."
        )

        if cutoff > training_cutoff:
            st.warning(
                "The uploaded data extends beyond the model's training "
                "cutoff. The saved model can score these records, but "
                "prediction quality on later periods must be validated."
            )

        # Features are calculated only from rows with both a usable
        # customer identifier and date, up to the scoring cutoff.
        history = raw.loc[
            raw[DATE].notna()
            & raw[MOBILE].notna()
            & (raw[DATE] <= cutoff)
        ].copy()

        if history.empty:
            st.error(
                "No rows with both valid dates and mobile numbers "
                "are available for customer-level scoring."
            )
            st.stop()

        # Match the historical training feature definitions.
        last_30_start = cutoff - pd.Timedelta(days=30)
        last_60_start = cutoff - pd.Timedelta(days=60)
        last_90_start = cutoff - pd.Timedelta(days=90)

        history["_last30"] = (
            history[DATE] > last_30_start
        ).astype("int8")

        history["_previous30"] = (
            (history[DATE] > last_60_start)
            & (history[DATE] <= last_30_start)
        ).astype("int8")

        history["_last60"] = (
            history[DATE] > last_60_start
        ).astype("int8")

        history["_last90"] = (
            history[DATE] > last_90_start
        ).astype("int8")

        grouped = history.groupby(MOBILE, sort=False)

        features = grouped.agg(
            Total_Appointments=(DATE, "size"),
            First_Appointment_Date=(DATE, "min"),
            Last_Appointment_Date=(DATE, "max"),
            Unique_Hospitals=("Hospital", "nunique"),
            Unique_Doctors=("Doctor", "nunique"),
            Unique_Specializations=("Specialization", "nunique"),
            Unique_Channels=("Channel", "nunique"),
            Unique_Towns=("Town", "nunique"),
        )

        window_counts = grouped[
            ["_last30", "_previous30", "_last60", "_last90"]
        ].sum()

        window_counts = window_counts.rename(
            columns={
                "_last30": "Appointments_Last_30_Days",
                "_previous30": "Appointments_Previous_30_Days",
                "_last60": "Appointments_Last_60_Days",
                "_last90": "Appointments_Last_90_Days",
            }
        )

        features = features.join(window_counts)

        fee_stats = grouped[FEE].agg(["sum", "mean", "count"])
        fee_stats = fee_stats.rename(
            columns={
                "sum": "Total_ECH_Fee",
                "mean": "Average_ECH_Fee",
                "count": "Fee_Records_Count",
            }
        )

        # Match sum(min_count=1): all-missing fees remain missing.
        fee_stats.loc[
            fee_stats["Fee_Records_Count"] == 0,
            "Total_ECH_Fee",
        ] = np.nan

        features = features.join(fee_stats)

        features["Days_Since_Last_Appointment"] = (
            cutoff - features["Last_Appointment_Date"]
        ).dt.days

        features["Customer_History_Days"] = (
            cutoff - features["First_Appointment_Date"]
        ).dt.days

        features["Snapshot_Date"] = cutoff
        features.index.name = MOBILE
        features = features.reset_index()

        # Match the exact feature order saved with the trained model.
        for col in feature_columns:
            if col not in features.columns:
                features[col] = np.nan
            features[col] = pd.to_numeric(
                features[col],
                errors="coerce",
            )

        st.subheader("2. Calculate predictions")

        st.write(
            f"Customers available for scoring: "
            f"**{len(features):,}**"
        )

        if st.button("Predict customer risk", type="primary"):
            X = features[feature_columns]

            probabilities = model.predict_proba(X)
            classes = list(model.classes_)

            if "Lost" not in classes or "Recurring" not in classes:
                st.error(
                    "The saved model does not contain the expected "
                    "Lost and Recurring classes."
                )
                st.stop()

            lost_idx = classes.index("Lost")
            recurring_idx = classes.index("Recurring")

            results = features[
                [MOBILE, "Snapshot_Date"]
            ].copy()

            results["Lost_Probability"] = probabilities[:, lost_idx]
            results["Recurring_Probability"] = (
                probabilities[:, recurring_idx]
            )

            results["Predicted_Status"] = np.where(
                results["Lost_Probability"] >= 0.5,
                "Lost",
                "Recurring",
            )

            results["Risk_Level"] = np.select(
                [
                    results["Lost_Probability"] >= 0.70,
                    results["Lost_Probability"] >= 0.40,
                ],
                ["High", "Medium"],
                default="Low",
            )

            st.session_state["customer_predictions"] = results
            st.session_state["customer_features"] = features
            st.session_state["scoring_cutoff"] = str(cutoff.date())

        if "customer_predictions" in st.session_state:
            results = st.session_state["customer_predictions"].copy()

            # Avoid displaying predictions from a previous upload.
            if st.session_state.get("scoring_cutoff") != str(cutoff.date()):
                st.session_state.pop("customer_predictions", None)
                st.session_state.pop("customer_features", None)
                st.info("Click Predict customer risk to score this upload.")
                st.stop()

            st.subheader("3. Prediction summary")

            total = len(results)
            lost_count = int(
                (results["Predicted_Status"] == "Lost").sum()
            )
            recurring_count = int(
                (results["Predicted_Status"] == "Recurring").sum()
            )
            high_count = int(
                (results["Risk_Level"] == "High").sum()
            )

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Customers scored", f"{total:,}")
            c2.metric("Predicted Lost", f"{lost_count:,}")
            c3.metric("Predicted Recurring", f"{recurring_count:,}")
            c4.metric("High risk", f"{high_count:,}")

            risk_counts = (
                results["Risk_Level"]
                .value_counts()
                .reindex(["High", "Medium", "Low"], fill_value=0)
            )

            st.write("**Risk level distribution**")
            st.bar_chart(risk_counts)

            st.subheader("4. Filter predictions")

            risk_filter = st.multiselect(
                "Risk level",
                ["High", "Medium", "Low"],
                default=["High", "Medium", "Low"],
            )

            status_filter = st.multiselect(
                "Predicted status",
                ["Lost", "Recurring"],
                default=["Lost", "Recurring"],
            )

            filtered = results.loc[
                results["Risk_Level"].isin(risk_filter)
                & results["Predicted_Status"].isin(status_filter)
            ].copy()

            st.write(f"Matching customers: **{len(filtered):,}**")

            display = filtered.copy()
            display["Lost_Probability"] = (
                display["Lost_Probability"] * 100
            ).round(2)
            display["Recurring_Probability"] = (
                display["Recurring_Probability"] * 100
            ).round(2)

            st.dataframe(display, use_container_width=True)

            st.subheader("5. Download predictions")

            include_full_numbers = st.checkbox(
                "Include full mobile numbers in downloaded CSV",
                value=False,
            )

            export = filtered.copy()

            if not include_full_numbers:
                export[MOBILE] = export[MOBILE].astype(str).apply(
                    lambda x: "*" * max(0, len(x) - 4) + x[-4:]
                )

            export["Lost_Probability"] = (
                export["Lost_Probability"] * 100
            ).round(2)

            export["Recurring_Probability"] = (
                export["Recurring_Probability"] * 100
            ).round(2)

            st.download_button(
                "Download predictions CSV",
                data=export.to_csv(
                    index=False
                ).encode("utf-8-sig"),
                file_name="customer_risk_predictions.csv",
                mime="text/csv",
            )

            st.caption(
                "Probabilities are model estimates, not confirmed "
                "customer statuses. The 50% classification threshold "
                "and risk bands are defined in the app."
            )

        st.subheader("6. Records excluded from customer scoring")

        no_mobile = raw.loc[
            raw[MOBILE].isna() | raw[MOBILE].eq("")
        ]

        st.write(
            f"Appointment rows without a usable mobile number: "
            f"**{len(no_mobile):,}**. These rows cannot be assigned "
            "to a specific customer by this app."
        )

        if not no_mobile.empty:
            st.download_button(
                "Download rows without mobile numbers",
                data=no_mobile.to_csv(
                    index=False
                ).encode("utf-8-sig"),
                file_name="appointments_without_mobile.csv",
                mime="text/csv",
            )

    except Exception as exc:
        st.error(f"Unable to process this CSV: {exc}")
