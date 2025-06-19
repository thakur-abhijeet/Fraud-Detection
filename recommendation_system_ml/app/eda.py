import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def load_data(path):
    return pd.read_csv(path)

def plot_distributions(df):
    plt.figure(figsize=(10, 4))
    sns.histplot(df["price"], bins=50, kde=True)
    plt.title("Price Distribution")
    plt.show()

    plt.figure(figsize=(10, 4))
    sns.countplot(data=df, x="category", order=df["category"].value_counts().index[:10])
    plt.title("Top Categories")
    plt.xticks(rotation=45)
    plt.show()

    plt.figure(figsize=(10, 4))
    sns.countplot(data=df, x="brand", order=df["brand"].value_counts().index[:10])
    plt.title("Top Brands")
    plt.xticks(rotation=45)
    plt.show()

def show_rating_distribution(df):
    sns.boxplot(data=df, x="category", y="rating")
    plt.title("Rating distribution per category")
    plt.xticks(rotation=45)
    plt.show()

if __name__ == "__main__":
    df = load_data("data/products.csv")
    plot_distributions(df)
    show_rating_distribution(df)
