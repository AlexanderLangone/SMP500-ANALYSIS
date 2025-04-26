
import pandas as pd
import numpy as np

def calculate_signals(df, ma_short=50, ma_long=200, rsi_period=14, rsi_upper=70, rsi_lower=30):
    """Calculate trading signals without future data leakage"""
    df = df.copy()
    
    # Calculate moving averages if parameters provided
    if ma_short is not None and ma_long is not None:
        df['MA_Short'] = df['Close'].rolling(window=ma_short).mean()
        df['MA_Long'] = df['Close'].rolling(window=ma_long).mean()
    
    # Calculate RSI if period provided
    if rsi_period is not None:
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
    
    # Generate signals
    df['Signal'] = 0  # 1 for buy, -1 for sell, 0 for hold
    
    if ma_short is not None and ma_long is not None:
        # MA Crossover only
        if rsi_period is None:
            df['Signal'] = np.where(df['MA_Short'] > df['MA_Long'], 1, -1)
        # Combined MA and RSI
        else:
            df['Signal'] = np.where((df['MA_Short'] > df['MA_Long']) & 
                                   (df['RSI'] < rsi_upper), 1, df['Signal'])
            df['Signal'] = np.where((df['MA_Short'] < df['MA_Long']) & 
                                   (df['RSI'] > rsi_lower), -1, df['Signal'])
    elif rsi_period is not None:
        # RSI only
        df['Signal'] = np.where(df['RSI'] < rsi_lower, 1, 
                               np.where(df['RSI'] > rsi_upper, -1, 0))
    
    return df

def backtest_strategy(df, strategy_name, initial_capital=100000):
    """Backtest the strategy"""
    df = df.copy()
    
    # Initialize positions and portfolio value
    df['Position'] = df['Signal'].shift(1)  # Shift to avoid look-ahead bias
    df['Position'] = df['Position'].fillna(0)
    
    # Calculate returns
    df['Strategy_Return'] = df['Position'] * df['Close'].pct_change()
    df['Buy_Hold_Return'] = df['Close'].pct_change()
    
    # Calculate cumulative returns
    df['Strategy_Cum_Return'] = (1 + df['Strategy_Return']).cumprod()
    df['Buy_Hold_Cum_Return'] = (1 + df['Buy_Hold_Return']).cumprod()
    
    # Calculate portfolio value
    df['Portfolio_Value'] = initial_capital * df['Strategy_Cum_Return']
    
    # Calculate metrics
    total_trades = len(df[df['Signal'] != 0])
    winning_trades = len(df[df['Strategy_Return'] > 0])
    
    # Calculate average holding time
    position_changes = df['Position'].diff()
    trade_starts = df[position_changes != 0].index
    if len(trade_starts) > 1:
        holding_times = [(end - start).days for start, end in zip(trade_starts[:-1], trade_starts[1:])]
        avg_hold_time = sum(holding_times) / len(holding_times) if holding_times else 0
    else:
        avg_hold_time = 0
    
    strategy_return = df['Strategy_Cum_Return'].iloc[-1] - 1
    buy_hold_return = df['Buy_Hold_Cum_Return'].iloc[-1] - 1
    
    return {
        'Strategy_Name': strategy_name,
        'Strategy_Return': strategy_return * 100,
        'Buy_Hold_Return': buy_hold_return * 100,
        'Total_Trades': total_trades,
        'Win_Rate': (winning_trades / total_trades * 100) if total_trades > 0 else 0,
        'Avg_Hold_Time': avg_hold_time,
        'Final_Portfolio': df['Portfolio_Value'].iloc[-1]
    }



def calculate_bollinger_signals(df, window=20, std_dev=2):
    """Calculate Bollinger Bands signals"""
    df = df.copy()
    df['MA'] = df['Close'].rolling(window=window).mean()
    df['STD'] = df['Close'].rolling(window=window).std()
    df['Upper'] = df['MA'] + (df['STD'] * std_dev)
    df['Lower'] = df['MA'] - (df['STD'] * std_dev)
    
    df['Signal'] = 0
    df['Signal'] = np.where(df['Close'] < df['Lower'], 1, df['Signal'])
    df['Signal'] = np.where(df['Close'] > df['Upper'], -1, df['Signal'])
    return df

def calculate_momentum_signals(df, rsi_period=14, fast=12, slow=26, signal=9):
    """Calculate combined RSI and MACD signals"""
    df = df.copy()
    # Calculate RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Calculate MACD
    exp1 = df['Close'].ewm(span=fast).mean()
    exp2 = df['Close'].ewm(span=slow).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=signal).mean()
    
    df['Signal'] = 0
    # Buy when RSI < 30 and MACD crosses above Signal Line
    df['Signal'] = np.where((df['RSI'] < 30) & (df['MACD'] > df['Signal_Line']), 1, df['Signal'])
    # Sell when RSI > 70 and MACD crosses below Signal Line
    df['Signal'] = np.where((df['RSI'] > 70) & (df['MACD'] < df['Signal_Line']), -1, df['Signal'])
    return df

def calculate_triple_ma_signals(df, short=20, medium=50, long=200):
    """Calculate Triple Moving Average signals"""
    df = df.copy()
    df['MA_Short'] = df['Close'].rolling(window=short).mean()
    df['MA_Medium'] = df['Close'].rolling(window=medium).mean()
    df['MA_Long'] = df['Close'].rolling(window=long).mean()
    
    df['Signal'] = 0
    # Buy when short > medium > long
    df['Signal'] = np.where((df['MA_Short'] > df['MA_Medium']) & 
                           (df['MA_Medium'] > df['MA_Long']), 1, df['Signal'])
    # Sell when short < medium < long
    df['Signal'] = np.where((df['MA_Short'] < df['MA_Medium']) & 
                           (df['MA_Medium'] < df['MA_Long']), -1, df['Signal'])
    return df
