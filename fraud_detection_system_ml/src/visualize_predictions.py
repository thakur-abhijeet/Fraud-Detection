import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load predictions
df = pd.read_csv("../models/predictions.csv")  # Change path if needed
print("📄 Loaded columns:", df.columns.tolist())

# Setup
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)
os.makedirs("visualizations", exist_ok=True)

### 1. Fraud Prediction Count
plt.figure()
sns.countplot(x="predicted_fraud", data=df)
plt.title("Count of Predicted Fraud vs Non-Fraud")
plt.xticks([0, 1], ["Not Fraud", "Fraud"])
plt.xlabel("Prediction")
plt.ylabel("Number of Transactions")
plt.tight_layout()
plt.savefig("visualizations/prediction_count.png")

### 2. Actual vs Predicted Fraud Comparison
plt.figure()
sns.countplot(x="is_fraud", hue="predicted_fraud", data=df)
plt.title("Actual Fraud vs Predicted Fraud")
plt.xticks([0, 1], ["Not Fraud", "Fraud"])
plt.xlabel("Actual Label")
plt.ylabel("Count")
plt.legend(title="Predicted")
plt.tight_layout()
plt.savefig("visualizations/actual_vs_predicted.png")

### 3. Fraud Probability Distribution
plt.figure()
sns.histplot([df.get("fraud_probability")], bins=30, kde=True)
plt.title("Fraud Probability Distribution")
plt.xlabel("Predicted Fraud Probability")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("visualizations/fraud_probability_distribution.png")

### 4. Amount vs Fraud Probability
plt.figure()
sns.scatterplot(
    x="amount",
    y="fraud_probability",
    hue="predicted_fraud",
    data=df,
    alpha=0.6,
    palette="coolwarm",
)
plt.title("Transaction Amount vs Fraud Probability")
plt.xlabel("Amount")
plt.ylabel("Fraud Probability")
plt.tight_layout()
plt.savefig("visualizations/amount_vs_fraud_probability.png")

### 5. High-Risk Transactions Table (Top 10)
top_frauds = df.sort_values(by="fraud_probability", ascending=False).head(10)
print("\n🔍 Top 10 High-Risk Transactions:")
print(
    top_frauds[
        [
            "transaction_id",
            "customer_id",
            "amount",
            "fraud_probability",
            "predicted_fraud",
        ]
    ]
)

### 6. Fraud Probability by Payment Method
plt.figure()

# Drop rows with missing or invalid values for boxplot
filtered_df = df[["payment_method", "fraud_probability"]].dropna()

# Ensure at least 2 valid values per category
filtered_df = filtered_df.groupby("payment_method").filter(lambda x: len(x) > 1)

# Only plot if there’s valid data
if not filtered_df.empty:
    plt.figure()
    sns.boxplot(x="payment_method", y="fraud_probability", data=filtered_df)
    plt.title("Fraud Probability by Payment Method")
    plt.xlabel("Payment Method")
    plt.ylabel("Fraud Probability")
    plt.tight_layout()
    plt.savefig("visualizations/fraud_by_payment_method.png")
else:
    print("⚠️ Not enough data per payment method for boxplot.")
plt.title("Fraud Probability by Payment Method")
plt.xlabel("Payment Method")
plt.ylabel("Fraud Probability")
plt.tight_layout()
plt.savefig("visualizations/fraud_by_payment_method.png")

### 7. Fraud Over Day of Week
plt.figure()
sns.barplot(x="timestamp_weekday", y="fraud_probability", data=df)
plt.title("Average Fraud Probability by Day of Week")
plt.xlabel("Day of Week (0=Mon, 6=Sun)")
plt.ylabel("Avg Fraud Probability")
plt.tight_layout()
plt.savefig("visualizations/fraud_by_day.png")

# Optionally display all at the end (if interactive)
# plt.show()
