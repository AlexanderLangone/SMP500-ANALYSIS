
import pandas as pd
import matplotlib.pyplot as plt

def calculate_buffett_indicator(sp500_df, gdp_df, market_cap_df):
    merged = pd.concat([sp500_df['Close'], gdp_df['GDP'], market_cap_df['MarketCap']], axis=1)
    merged.columns = ['S&P500', 'GDP', 'MarketCap']
    merged.dropna(inplace=True)
    merged['BuffettIndicator'] = merged['MarketCap'] / merged['GDP']
    return merged

def plot_buffett_indicator(merged_df, save_path="plots/buffett_indicator_trend.png"):
    merged_df['BuffettIndicator'].plot(title='Buffett Indicator Over Time')
    plt.ylabel('Market Cap / GDP')
    plt.savefig(save_path)
    print(f"Buffett Indicator plot saved to {save_path}")
