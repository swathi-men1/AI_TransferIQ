import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server use
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid")
plt.rcParams['figure.dpi'] = 100


def run_eda(data_path="transferiq_dataset.csv", out_dir="visualizations"):
    os.makedirs(out_dir, exist_ok=True)
    print(f"Running EDA on {data_path}...")
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])

    report = {}

    # ── 1. Market Value Distribution ──────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df['market_value'] / 1_000_000, bins=50, color='#818cf8', edgecolor='none', alpha=0.85)
    ax.set_title('Market Value Distribution', fontsize=16, fontweight='bold')
    ax.set_xlabel('Market Value (€M)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/eda_market_value_dist.png')
    plt.close()
    report['market_value_dist'] = f'{out_dir}/eda_market_value_dist.png'

    # ── 2. Performance Rating by Position ─────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    positions = df['position'].unique()
    for pos in positions:
        subset = df[df['position'] == pos]['performance_rating']
        ax.hist(subset, bins=30, alpha=0.65, label=pos)
    ax.set_title('Performance Rating Distribution by Position', fontsize=16, fontweight='bold')
    ax.set_xlabel('Performance Rating', fontsize=12)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{out_dir}/eda_perf_by_position.png')
    plt.close()
    report['perf_by_position'] = f'{out_dir}/eda_perf_by_position.png'

    # ── 3. Sentiment Score Distribution ───────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df['social_sentiment_score'], bins=40, color='#c084fc', edgecolor='none', alpha=0.85)
    ax.axvline(0, color='red', linestyle='--', linewidth=1, label='Neutral')
    ax.set_title('Social Sentiment Score Distribution', fontsize=16, fontweight='bold')
    ax.set_xlabel('Sentiment Score (-1 to 1)', fontsize=12)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{out_dir}/eda_sentiment_dist.png')
    plt.close()
    report['sentiment_dist'] = f'{out_dir}/eda_sentiment_dist.png'

    # ── 4. Correlation Heatmap ─────────────────────────────
    num_cols = ['performance_rating', 'goals_assists', 'minutes_played',
                'days_injured', 'social_sentiment_score', 'contract_duration_months', 'market_value']
    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax, square=True, linewidths=0.5)
    ax.set_title('Feature Correlation Heatmap', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{out_dir}/eda_correlation_heatmap.png')
    plt.close()
    report['correlation_heatmap'] = f'{out_dir}/eda_correlation_heatmap.png'

    # ── 5. Average Market Value Over Time ─────────────────
    monthly_avg = df.groupby('date')['market_value'].mean().reset_index()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(monthly_avg['date'], monthly_avg['market_value'] / 1_000_000, color='#818cf8', linewidth=2)
    ax.fill_between(monthly_avg['date'], monthly_avg['market_value'] / 1_000_000, alpha=0.2, color='#818cf8')
    ax.set_title('Average Market Value Over Time', fontsize=16, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Average Market Value (€M)', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/eda_value_over_time.png')
    plt.close()
    report['value_over_time'] = f'{out_dir}/eda_value_over_time.png'

    # ── Summary Stats ──────────────────────────────────────
    stats = df[num_cols].describe().round(2).to_dict()
    report['summary_stats'] = stats
    report['shape'] = {'rows': len(df), 'columns': len(df.columns)}
    report['missing_values'] = df.isnull().sum().to_dict()

    print("EDA complete. Visualizations saved.")
    return report


if __name__ == "__main__":
    run_eda()
