# main.py
"""
Stock Behavior Clustering using K-Means
Run: python main.py
Outputs saved in ./output
"""

import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def download_data(stocks, start="2023-01-01", end="2025-01-01"):
    print("Downloading data for:", stocks)
    df = yf.download(stocks, start=start, end=end)
    if 'Adj Close' in df:
        return df['Adj Close']
    # If single ticker, yf returns a Series; ensure DataFrame
    return df

def compute_features(df):
    # daily percentage returns
    returns = df.pct_change().dropna()
    features = pd.DataFrame({
        'Mean Return': returns.mean(),
        'Volatility': returns.std()
    })
    return features, returns

def plot_elbow(X_scaled, save_to):
    inertias = []
    Ks = range(1, 10)
    for k in Ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)
    plt.figure()
    plt.plot(list(Ks), inertias, marker='o')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_to)
    plt.close()
    print("Saved:", save_to)

def cluster_and_save(features, n_clusters=4, out_dir='output'):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    # elbow plot
    plot_elbow(X_scaled, os.path.join(out_dir, 'elbow.png'))

    # KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    features['Cluster'] = kmeans.fit_predict(X_scaled)

    # save CSV
    csv_path = os.path.join(out_dir, 'stock_clusters.csv')
    features.to_csv(csv_path)
    print("Saved:", csv_path)

    # cluster scatter
    plt.figure(figsize=(10,6))
    sns.scatterplot(x='Volatility', y='Mean Return', data=features, hue='Cluster', palette='tab10', s=120)
    # annotate points
    x_off = (features['Volatility'].max() - features['Volatility'].min()) * 0.01
    y_off = (features['Mean Return'].max() - features['Mean Return'].min()) * 0.01
    for ticker in features.index:
        plt.text(features.loc[ticker, 'Volatility'] + x_off,
                 features.loc[ticker, 'Mean Return'] + y_off,
                 ticker, fontsize=9)
    plt.title(f'Stock clusters (k={n_clusters})')
    plt.xlabel('Volatility (std of returns)')
    plt.ylabel('Mean Return')
    plt.tight_layout()
    cluster_png = os.path.join(out_dir, 'clusters.png')
    plt.savefig(cluster_png)
    plt.close()
    print("Saved:", cluster_png)

def correlation_heatmap(returns, out_dir='output'):
    plt.figure(figsize=(10,8))
    sns.heatmap(returns.corr(), annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)
    plt.title('Return Correlation Heatmap')
    plt.tight_layout()
    heatmap_png = os.path.join(out_dir, 'corr.png')
    plt.savefig(heatmap_png)
    plt.close()
    print("Saved:", heatmap_png)

def main():
    out_dir = 'output'
    os.makedirs(out_dir, exist_ok=True)

    # Choose your stocks (you can edit this list)
    stocks = ['AAPL','MSFT','GOOGL','AMZN','META','TSLA','NVDA','NFLX','INTC','IBM','ORCL','AMD']

    # Download
    prices = download_data(stocks, start="2023-01-01", end="2025-01-01")
    if prices is None or prices.empty:
        print("No data downloaded. Check your internet connection or ticker symbols.")
        return

    # Features & clustering
    features, returns = compute_features(prices)
    cluster_and_save(features, n_clusters=4, out_dir=out_dir)
    correlation_heatmap(returns, out_dir=out_dir)

    print("\nRESULTS (first rows):")
    print(features.head())

if __name__ == "__main__":
    main()
