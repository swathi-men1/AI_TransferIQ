import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
 
class EDA:
    def __init__(self,df):
        self.df=df
    
    @staticmethod
    def eda(input_file="D:\\New folder (5)\\infosys\\AI_TransferIQ\\data\\transferiq_dataset.csv",output_path="D:\\New folder (5)\\infosys\\AI_TransferIQ\\reports\\figures"):
        os.makedirs(output_path,exist_ok=True)
        df=pd.read_csv(input_file)
        df['date']=pd.to_datetime(df['date'])
        reports={}

        figure,ax=plt.subplots(figsize=(12,10))
        ax.hist(df['market_value'] / 1_000_000, bins=50, color='#818cf8', edgecolor='none', alpha=0.85)
        ax.set_title('Market Value Distribution', fontsize=16, fontweight='bold')
        ax.set_xlabel('Market Value (€M)', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        plt.tight_layout()
        plt.savefig(f'{output_path}/eda_market_value.png')
        plt.close()
        reports['market_value_dist'] = f'{output_path}/eda_market_value.png'
        
        fig, ax = plt.subplots(figsize=(10, 5))
        positions = df['position'].unique()
        for pos in positions:
            subset = df[df['position'] == pos]['performance_rating']
            ax.hist(subset, bins=30, alpha=0.65, label=pos)
        ax.set_title('Performance Rating Distribution by Position', fontsize=16, fontweight='bold')
        ax.set_xlabel('Performance Rating', fontsize=12)
        ax.legend()
        plt.tight_layout()
        plt.savefig(f'{output_path}/eda_perf_by_position.png')
        plt.close()
        reports['perf_by_position'] = f'{output_path}/eda_perf_by_position.png'
    
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(df['social_sentiment_score'], bins=40, color='#c084fc', edgecolor='none', alpha=0.85)
        ax.axvline(0, color='red', linestyle='--', linewidth=1, label='Neutral')
        ax.set_title('Social Sentiment Score Distribution', fontsize=16, fontweight='bold')
        ax.set_xlabel('Sentiment Score (-1 to 1)', fontsize=12)
        ax.legend()
        plt.tight_layout()
        plt.savefig(f'{output_path}/eda_sentiment_dist.png')
        plt.close()
        reports['sentiment_dist'] = f'{output_path}/eda_sentiment_dist.png'

        num_cols = ['performance_rating', 'goals_assists', 'minutes_played',
                'days_injured', 'social_sentiment_score', 'contract_duration_months', 'market_value']
        corr = df[num_cols].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax, square=True, linewidths=0.5)
        ax.set_title('Feature Correlation Heatmap', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{output_path}/eda_correlation_heatmap.png')
        plt.close()
        reports['correlation_heatmap'] = f'{output_path}/eda_correlation_heatmap.png'

        monthly_avg = df.groupby('date')['market_value'].mean().reset_index()
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(monthly_avg['date'], monthly_avg['market_value'] / 1_000_000, color='#818cf8', linewidth=2)
        ax.fill_between(monthly_avg['date'], monthly_avg['market_value'] / 1_000_000, alpha=0.2, color='#818cf8')
        ax.set_title('Average Market Value Over Time', fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Average Market Value (€M)', fontsize=12)
        plt.tight_layout()
        plt.savefig(f'{output_path}/eda_value_over_time.png')
        plt.close()
        reports['value_over_time'] = f'{output_path}/eda_value_over_time.png'

        stats = df[num_cols].describe().round(2).to_dict()
        reports['summary_stats'] = stats
        reports['shape'] = {'rows': len(df), 'columns': len(df.columns)}
        reports['missing_values'] = df.isnull().sum().to_dict()

        print("EDA complete. Visualizations saved.")
        return reports


if __name__ == "__main__":
    EDA.eda()
