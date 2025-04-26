
import pandas as pd

def load_gdp_data(filepath="data/gdp.csv"):
    gdp = pd.read_csv(filepath, parse_dates=True, index_col=0)
    return gdp

def resample_to_annual(df, column='Close'):
    return df[column].resample('Y').mean()

def compute_correlation(sp500_annual, gdp_annual):
    combined = pd.concat([sp500_annual, gdp_annual], axis=1).dropna()
    correlation = combined.corr().iloc[0, 1]
    return correlation
