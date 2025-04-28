
import yfinance as yf
import pandas as pd
from pandas_datareader import data as pdr
from contextlib import redirect_stdout
import io

def get_valid_date(prompt, min_date, max_date=None):
    while True:
        date_str = input(prompt).strip()
        try:
            date = pd.to_datetime(date_str)
            if date < pd.to_datetime(min_date):
                print(f"Date must be after {min_date}")
                continue
            if max_date and date > pd.to_datetime(max_date):
                print(f"Date must be before {max_date}")
                continue
            return date
        except ValueError:
            print("Invalid date format. Please use YYYY-MM-DD")

def download_sp500_data(filepath='data/sp500.csv'):
    # Download regular SP500 (no dividends) from 1927
    try:
        buf = io.StringIO()
        with redirect_stdout(buf):
            sp500_regular = yf.download('^GSPC', auto_adjust=True, progress=False)  # Regular S&P 500 index from 1927
        if not sp500_regular.empty:
            sp500_regular.to_csv('data/sp500_no_div.csv')
    except Exception as e:
        print(f"Error downloading regular SP500 data: {e}")
        return

    # Download SP500TR (with dividends) from 1988
    try:
        buf = io.StringIO()
        with redirect_stdout(buf):
            sp500_tr = yf.download('^SP500TR', start='1988-01-01', progress=False)  # Total Return version
        if not sp500_tr.empty:
            sp500_tr.to_csv('data/sp500_with_div.csv')
            print("\nAll up to date Data has been downloaded\n")
    except Exception as e:
        print(f"Error downloading SP500TR data: {e}")
        return

    # Ask user which dataset to use for analysis
    while True:
        choice = input("Which dataset would you like to use for analysis? (dividends/no-dividends): ").lower().strip()
        if choice in ['dividends', 'div', 'd']:
            sp500_tr.to_csv(filepath)
            print(f"Using dividend-adjusted data for analysis (data from 1988-01-04)")
            with open('main.py', 'r') as file:
                content = file.read()
            if '1927-12-30' in content:
                content = content.replace('_default_start = "1927-12-30"', '_default_start = "1988-01-04"')
                content = content.replace('press Enter for 1927-12-30', 'press Enter for 1988-01-04')
                content = content.replace('Enter start date (press Enter for 1927-12-30', 'Enter start date (press Enter for 1988-01-04')
                with open('main.py', 'w') as file:
                    file.write(content)
            return
        elif choice in ['no-dividends', 'no-div', 'n']:
            sp500_regular.to_csv(filepath)
            print(f"Using non-dividend-adjusted data for analysis (data from 1927-12-30)")
            with open('main.py', 'r') as file:
                content = file.read()
            if choice in ['no-dividends', 'no-div', 'n']:
                content = content.replace('_default_start = "1988-01-04"', '_default_start = "1927-12-30"')
                content = content.replace('press Enter for 1988-01-04', 'press Enter for 1927-12-30')
                content = content.replace('Enter start date (press Enter for 1988-01-04', 'Enter start date (press Enter for 1927-12-30')
                with open('main.py', 'w') as file:
                    file.write(content)
            return
        else:
            print("Please answer 'dividends' or 'no-dividends'")

def download_gdp_data(filepath='data/gdp.csv'):
    try:
        gdp = pdr.DataReader('GDP', 'fred', start='1947-01-01')
        gdp.to_csv(filepath)
        print(f"GDP data saved to {filepath}")
    except Exception as e:
        print(f"Error downloading GDP data: {e}")
        raise

def download_recession_data(filepath='data/recession.csv'):
    try:
        rec = pdr.DataReader('USREC', 'fred', start='1947-01-01')
        rec.to_csv(filepath)
        print(f"Recession data saved to {filepath}")
    except Exception as e:
        print(f"Error downloading recession data: {e}")
        raise
