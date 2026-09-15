# Machine-Learning-Practice
Customer Churn Analysis project using Python, Machine Learning, K-Means++ clustering, PCA, and data visualization techniques to identify customer retention patterns and business insights.
tomer Churn Data Analysis
Overview and tenure (time) analysis for Data_for_ML.csv

Usage:
    python analysis.py --input Data_for_ML.csv --outdir images
"""

import argparse
import os

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def print_overview(df: pd.DataFrame) -> None:
    print("=== Overview ===")
    print(f"Rows: {len(df)}, Columns: {df.shape[1]}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print(f"Churn rate: {df['churn'].mean():.2%}\n")

    print("=== Categorical distributions ===")
    cat_cols = [
        "contract_type", "tech_support", "internet_service",
        "payment_method", "paperless_billing", "has_partner",
    ]
    for col in cat_cols:
        print(f"{col}: {dict(df[col].value_counts())}")
    print()

    print("=== Correlation with churn ===")
    num_cols = [
        "tenure_months", "monthly_charges", "total_charges",
        "support_calls_last_year", "senior_citizen", "churn",
    ]
    print(df[num_cols].corr()["churn"].sort_values(ascending=False))
    print()


def tenure_analysis(df: pd.DataFrame, outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)

    bins = [-1, 6, 12, 24, 36, 48, 60, 72]
    labels = ["0-6", "7-12", "13-24", "25-36", "37-48", "49-60", "61-72"]
    df["tenure_bucket"] = pd.cut(df["tenure_months"], bins=bins, labels=labels)

    print("=== Tenure by churn status ===")
    print(df.groupby("churn")["tenure_months"].describe())
    print()

    churn_by_tenure = df.groupby("tenure_bucket", observed=True)["churn"].mean() * 100
    print("=== Churn rate by tenure bucket (%) ===")
    print(churn_by_tenure)
    print()
    ]
    print(df[num_cols].corr()["churn"].sort_values(ascending=False))
    print()


def tenure_analysis(df: pd.DataFrame, outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)

    bins = [-1, 6, 12, 24, 36, 48, 60, 72]
    labels = ["0-6", "7-12", "13-24", "25-36", "37-48", "49-60", "61-72"]
    df["tenure_bucket"] = pd.cut(df["tenure_months"], bins=bins, labels=labels)

    print("=== Tenure by churn status ===")
    print(df.groupby("churn")["tenure_months"].describe())
    print()

    churn_by_tenure = df.groupby("tenure_bucket", observed=True)["churn"].mean() * 100
    print("=== Churn rate by tenure bucket (%) ===")
    print(churn_by_tenure)
    print()

    churn_by_contract = df.groupby("contract_type")["churn"].mean() * 100
    tenure_by_contract = df.groupby("contract_type")["tenure_months"].mean()
    print("=== Churn rate & avg tenure by contract type ===")
    print(pd.DataFrame({
        "churn_rate_%": churn_by_contract,
        "avg_tenure_months": tenure_by_contract,
    }))
    print()

    # Chart 1: churn rate by tenure bucket
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(churn_by_tenure.index.astype(str), churn_by_tenure.values, color="#4C72B0")
    ax.set_xlabel("Tenure (months)")
    ax.set_ylabel("Churn rate (%)")
    ax.set_title("Churn Rate by Tenure Bucket")
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1,
                 f"{b.get_height():.1f}%", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "churn_by_tenure.png"), dpi=150)
    plt.close()

    # Chart 2: churn rate by contract type
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(churn_by_contract.index, churn_by_contract.values, color="#DD8452")
    ax.set_xlabel("Contract Type")
    ax.set_ylabel("Churn rate (%)")
ax.set_title("Churn Rate by Contract Type")
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1,
                 f"{b.get_height():.1f}%", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "churn_by_contract.png"), dpi=150)
    plt.close()

    # Chart 3: tenure distribution split by churn
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df[df.churn == 0]["tenure_months"], bins=24, alpha=0.6, label="Retained", color="#55A868")
    ax.hist(df[df.churn == 1]["tenure_months"], bins=24, alpha=0.6, label="Churned", color="#C44E52")
    ax.set_xlabel("Tenure (months)")
    ax.set_ylabel("Number of customers")
    ax.set_title("Tenure Distribution: Churned vs Retained")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "tenure_distribution.png"), dpi=150)
    plt.close()

    print(f"Charts saved to: {outdir}/")


def main():
    parser = argparse.ArgumentParser(description="Overview + tenure (time) analysis of churn data")
    parser.add_argument("--input", default="Data_for_ML.csv", help="Path to input CSV")
    parser.add_argument("--outdir", default="images", help="Directory to save chart PNGs")
    args = parser.parse_args()

    df = load_data(args.input)
    print_overview(df)
    tenure_analysis(df, args.outdir)


if __name__ == "__main__":
    main()

