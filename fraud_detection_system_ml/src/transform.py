import pandas as pd
import numpy as np
import datetime
import sys


def align_dataset_to_model_format(df_raw):
    """
    Align any incoming dataset (same domain) to the target schema expected by the model.
    """

    # --- 1. Define the expected model schema ---
    target_columns = [
        "transaction_id",
        "customer_id",
        "product_id",
        "payment_method",
        "device_id",
        "amount",
        "is_fraud",
        "time_since_last_transaction",
        "transactions_past_1hr",
        "timestamp_hour",
        "timestamp_day",
        "timestamp_weekday",
        "prev_timestamp_hour",
        "prev_timestamp_day",
        "prev_timestamp_weekday",
    ]

    # --- 2. Rename common alternate column names ---
    rename_map = {
        "trans_id": "transaction_id",
        "cust_id": "customer_id",
        "user_id": "customer_id",
        "prod_id": "product_id",
        "pay_method": "payment_method",
        "device": "device_id",
        "amt": "amount",
        "fraud_flag": "is_fraud",
        "ts": "timestamp",
    }
    df = df_raw.rename(
        columns={k: v for k, v in rename_map.items() if k in df_raw.columns}
    )

    # --- 3. Ensure 'customer_id' exists ---
    if "customer_id" not in df.columns:
        print("⚠️ Warning: No 'customer_id' column found. Adding synthetic IDs.")
        df["customer_id"] = 0

    # --- 4. Ensure 'timestamp' exists ---
    if "timestamp" not in df.columns:
        print("⚠️ Warning: No 'timestamp' column found. Adding synthetic timestamps.")
        now = pd.Timestamp(datetime.datetime.now())
        df["timestamp"] = [now - pd.Timedelta(minutes=i) for i in range(len(df))][::-1]
    else:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        if df["timestamp"].isnull().any():
            print(
                "⚠️ Warning: Some timestamps couldn't be parsed. Filling with current time."
            )
            df["timestamp"] = df["timestamp"].fillna(
                pd.Timestamp(datetime.datetime.now())
            )

    # --- 5. Feature Engineering ---
    df = df.sort_values(["customer_id", "timestamp"]).reset_index(drop=True)

    df["prev_timestamp"] = df.groupby("customer_id")["timestamp"].shift(1)
    df["time_since_last_transaction"] = (
        (df["timestamp"] - df["prev_timestamp"]).dt.total_seconds().fillna(-1)
    )

    # Rolling transaction counts in past 1 hour per customer (excluding current)
    roll_col = "transaction_id" if "transaction_id" in df.columns else "amount"

    df_rolling = (
        df.set_index("timestamp")
        .groupby("customer_id")
        .rolling("1h")[roll_col]
        .count()
        .reset_index(name="transactions_past_1hr")
    )

    df_rolling["transactions_past_1hr"] -= 1  # Exclude current

    # Merge back
    df = pd.merge_asof(
        df.sort_values(["customer_id", "timestamp"]),
        df_rolling.sort_values(["customer_id", "timestamp"]),
        on="timestamp",
        by="customer_id",
        direction="backward",
    )

    df["transactions_past_1hr"] = df["transactions_past_1hr"].fillna(0)
    # Timestamp feature extraction
    df["timestamp_hour"] = df["timestamp"].dt.hour
    df["timestamp_day"] = df["timestamp"].dt.day
    df["timestamp_weekday"] = df["timestamp"].dt.weekday

    # Previous timestamp features
    df["prev_timestamp_hour"] = df["prev_timestamp"].dt.hour.fillna(-1).astype(int)
    df["prev_timestamp_day"] = df["prev_timestamp"].dt.day.fillna(-1).astype(int)
    df["prev_timestamp_weekday"] = (
        df["prev_timestamp"].dt.weekday.fillna(-1).astype(int)
    )

    # --- 6. Filter columns ---
    df_final = df[[col for col in target_columns if col in df.columns]].copy()

    # --- 7. Type Conversions ---
    dtype_map = {
        "transaction_id": int,
        "customer_id": int,
        "product_id": int,
        "payment_method": "category",
        "device_id": "category",
        "amount": float,
        "is_fraud": bool,
        "time_since_last_transaction": float,
        "transactions_past_1hr": float,
        "timestamp_hour": int,
        "timestamp_day": int,
        "timestamp_weekday": int,
        "prev_timestamp_hour": int,
        "prev_timestamp_day": int,
        "prev_timestamp_weekday": int,
    }

    for col, dtype in dtype_map.items():
        if col in df_final.columns:
            try:
                if dtype == "category":
                    df_final[col] = df_final[col].astype("category")
                else:
                    df_final[col] = df_final[col].astype(dtype)
            except Exception as e:
                print(f"⚠️ Warning: Could not convert column '{col}' to {dtype}: {e}")

    return df_final


# --- CLI Usage ---
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python transform.py path/to/dataset.csv")
        sys.exit(1)

    input_csv = sys.argv[1]
    try:
        df_raw = pd.read_csv(input_csv)
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        sys.exit(1)

    df_aligned = align_dataset_to_model_format(df_raw)

    output_csv = input_csv.replace(".csv", "_aligned.csv")
    df_aligned.to_csv(output_csv, index=False)
    print(f"✅ Aligned dataset saved to: {output_csv}")
