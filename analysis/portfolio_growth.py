import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def analyze_portfolio_growth(df,
                             initial_capital=100000,
                             save_dir='plots/portfolio'):
    os.makedirs(save_dir, exist_ok=True)

    def format_y_axis(ax):

        def currency_formatter(x, p):
            if abs(x) >= 1e6:
                return f'${x/1e6:.1f}M'
            elif abs(x) >= 1000:
                return f'${x/1000:.0f}K'
            return f'${x:.0f}'

        ax.yaxis.set_major_formatter(plt.FuncFormatter(currency_formatter))

    def get_axis_limits(values, initial_capital):
        min_val = values.min()
        max_val = values.max()
        padding = (max_val - min_val) * 0.1
        y_min = min(min_val - padding, initial_capital * -0.1)
        y_max = max_val + padding
        return y_min, y_max

    # Calculate all strategy portfolios
    # Buy & Hold Strategy
    df['Buy_Hold_Returns'] = df['Close'].pct_change()
    df['Buy_Hold_Value'] = initial_capital * (
        1 + df['Buy_Hold_Returns']).cumprod()
    df['Buy_Hold_Signal'] = 1  # Always in the market

    # Calculate Buy & Hold metrics
    buy_hold_trades = 1  # One trade (buy and hold)
    buy_hold_winning_trades = len(df[df['Buy_Hold_Returns'] > 0])
    buy_hold_avg_hold_time = (df.index[-1] - df.index[0]).days  # Entire period

    # Moving Average Crossover Strategy
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    df['MA_Signal'] = np.where(df['MA50'] > df['MA200'], 1, -1)
    df['MA_Returns'] = df['MA_Signal'].shift(1) * df['Close'].pct_change()
    df['MA_Portfolio'] = initial_capital * (1 + df['MA_Returns']).cumprod()

    # RSI Strategy
    def calculate_rsi(data, periods=14):
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    df['RSI'] = calculate_rsi(df['Close'])
    df['RSI_Signal'] = np.where(df['RSI'] < 30, 1,
                                np.where(df['RSI'] > 70, -1, 0))
    df['RSI_Returns'] = df['RSI_Signal'].shift(1) * df['Close'].pct_change()
    df['RSI_Portfolio'] = initial_capital * (1 + df['RSI_Returns']).cumprod()

    # Combined MA+RSI Strategy
    df['Combined_Signal'] = np.where(
        (df['MA_Signal'] == 1) & (df['RSI'] < 70), 1,
        np.where((df['MA_Signal'] == -1) & (df['RSI'] > 30), -1, 0))
    df['Combined_Returns'] = df['Combined_Signal'].shift(
        1) * df['Close'].pct_change()
    df['Combined_Portfolio'] = initial_capital * (
        1 + df['Combined_Returns']).cumprod()

    # Bollinger Bands Strategy
    window = 20
    df['BB_MA'] = df['Close'].rolling(window=window).mean()
    df['BB_STD'] = df['Close'].rolling(window=window).std()
    df['BB_Upper'] = df['BB_MA'] + (df['BB_STD'] * 2)
    df['BB_Lower'] = df['BB_MA'] - (df['BB_STD'] * 2)
    df['BB_Signal'] = np.where(df['Close'] < df['BB_Lower'], 1,
                               np.where(df['Close'] > df['BB_Upper'], -1, 0))
    df['BB_Returns'] = df['BB_Signal'].shift(1) * df['Close'].pct_change()
    df['BB_Portfolio'] = initial_capital * (1 + df['BB_Returns']).cumprod()

    # Momentum Strategy (RSI+MACD)
    exp1 = df['Close'].ewm(span=12).mean()
    exp2 = df['Close'].ewm(span=26).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9).mean()
    df['Momentum_Signal'] = np.where(
        (df['RSI'] < 30) & (df['MACD'] > df['Signal_Line']), 1,
        np.where((df['RSI'] > 70) & (df['MACD'] < df['Signal_Line']), -1, 0))
    df['Momentum_Returns'] = df['Momentum_Signal'].shift(
        1) * df['Close'].pct_change()
    df['Momentum_Portfolio'] = initial_capital * (
        1 + df['Momentum_Returns']).cumprod()

    # Triple MA Strategy
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    df['Triple_MA_Signal'] = np.where(
        (df['MA20'] > df['MA50']) & (df['MA50'] > df['MA200']), 1,
        np.where((df['MA20'] < df['MA50']) & (df['MA50'] < df['MA200']), -1,
                 0))
    df['Triple_MA_Returns'] = df['Triple_MA_Signal'].shift(
        1) * df['Close'].pct_change()
    df['Triple_MA_Portfolio'] = initial_capital * (
        1 + df['Triple_MA_Returns']).cumprod()

    # 12-Month Absolute Momentum (Cash) Strategy
    df['Price_12M_Ago'] = df['Close'].shift(
        252)  # Get price from exactly 1 year ago

    # Generate signals based on year-over-year price comparison
    # Buy (1) when current price > year ago price, Cash (0) when current price <= year ago price
    df['Abs_Mom_Cash_Signal'] = np.where(df['Close'] > df['Price_12M_Ago'], 1,
                                         0)

    # Calculate daily returns - when signal is 1 get market returns, when 0 get cash (0%)
    df['Abs_Mom_Cash_Returns'] = np.where(
        df['Abs_Mom_Cash_Signal'].shift(1) ==
        1,  # Shift signal to avoid look-ahead bias
        df['Close'].pct_change(),
        0)
    
    # Ensure first return is 0 to start at initial capital
    df.loc[df.index[0], 'Abs_Mom_Cash_Returns'] = 0

    # Calculate portfolio value
    df['Abs_Mom_Cash_Portfolio'] = initial_capital * (
        1 + df['Abs_Mom_Cash_Returns']).cumprod()

    # 12-MonthAbsolute Momentum (Bond) Strategy
    # Simplified version with fixed 2% annual return when in bonds
    df['Abs_Mom_Bond_Signal'] = np.where(df['Close'] > df['Price_12M_Ago'], 1,
                                         0)
    daily_bond_return = 0.02 / 252  # 2% annual return converted to daily
    df['Abs_Mom_Bond_Returns'] = np.where(
        df['Abs_Mom_Bond_Signal'].shift(1) == 1,
        df['Close'].pct_change(),  # Market returns when signal is 1
        daily_bond_return)  # Bond returns when signal is 0
    
    # Ensure first return is 0 to start at initial capital
    df.loc[df.index[0], 'Abs_Mom_Bond_Returns'] = 0
    df['Abs_Mom_Bond_Portfolio'] = initial_capital * (
        1 + df['Abs_Mom_Bond_Returns']).cumprod()

    # Dual Momentum Strategy with 1.25x Leverage
    # Calculate 12-month returns for both assets
    df['SP500_12M_Return'] = df['Close'].pct_change(252)
    df['Bond_12M_Return'] = daily_bond_return * 252  # Annualized bond return

    # Generate signals based on dual momentum rules
    df['Dual_Mom_Signal'] = pd.Series(0, index=df.index,
                                      dtype='float64')  # Initialize as float64
    # Invest in SP500 with leverage when it has positive momentum and beats bonds
    df.loc[(df['SP500_12M_Return'] > 0) &
           (df['SP500_12M_Return'] > df['Bond_12M_Return']),
           'Dual_Mom_Signal'] = 1.25
    # Invest in bonds when they have higher momentum
    df.loc[(df['Bond_12M_Return'] > df['SP500_12M_Return']),
           'Dual_Mom_Signal'] = 1

    # Calculate strategy returns
    df['Dual_Mom_Returns'] = np.where(
        df['Dual_Mom_Signal'].shift(1) == 1.25,
        df['Close'].pct_change() * 1.25,  # Leveraged SP500 returns
        np.where(
            df['Dual_Mom_Signal'].shift(1) == 1,
            daily_bond_return,  # Bond returns
            0  # Cash returns
        ))
    # Ensure first return is 0 to start at initial capital
    df.loc[df.index[0], 'Dual_Mom_Returns'] = 0
    df['Dual_Mom_Portfolio'] = initial_capital * (
        1 + df['Dual_Mom_Returns']).cumprod()

    # Plot Dual Momentum strategy
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), height_ratios=[2, 1])

    # Plot portfolio value
    ax1.plot(df.index, df['Dual_Mom_Portfolio'], label='Dual Momentum (1.25x)', color='purple')
    ax1.set_title('Dual Momentum (1.25x) Strategy Performance')
    ax1.set_ylabel('Portfolio Value')
    format_y_axis(ax1)
    
    # Calculate min/max based on portfolio values relative to initial capital
    port_min = df['Dual_Mom_Portfolio'].min()
    port_max = df['Dual_Mom_Portfolio'].max()
    padding = (port_max - port_min) * 0.1
    y_min = max(0, port_min - padding)  # Don't go below 0
    y_max = port_max + padding
    
    ax1.set_ylim(y_min, y_max)
    ax1.axhline(y=initial_capital, color='gray', linestyle='--', alpha=0.5, label='Initial Investment')
    ax1.legend(loc='upper left')
    ax1.grid(True)

    # Plot price and signals
    ax2.plot(df.index, df['Close'], color='gray', alpha=0.6, label='S&P 500')

    # Plot signals for state changes
    leverage_signals = df[df['Dual_Mom_Signal'] == 1.25].index  # Leveraged SP500
    bond_signals = df[df['Dual_Mom_Signal'] == 1].index  # Bonds
    cash_signals = df[df['Dual_Mom_Signal'] == 0].index  # Cash

    ax2.scatter(leverage_signals, df.loc[leverage_signals, 'Close'],
                color='green', marker='^', s=100, label='Enter Leveraged SP500')
    ax2.scatter(bond_signals, df.loc[bond_signals, 'Close'],
                color='blue', marker='v', s=100, label='Switch to Bonds')
    ax2.scatter(cash_signals, df.loc[cash_signals, 'Close'],
                color='red', marker='v', s=100, label='Switch to Cash')

    ax2.set_title('Strategy Signals')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('S&P 500 Price')
    ax2.legend(loc='upper left')
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/dual_momentum_performance.png')
    plt.close()

    # Calculate metrics for all strategies
    def calculate_strategy_metrics(signals, returns, portfolio):
        signal_changes = signals.diff().fillna(0)
        total_trades = len(signal_changes[signal_changes != 0])
        winning_trades = len(returns[returns > 0])

        # Calculate average holding time
        trade_starts = df[signal_changes != 0].index
        if len(trade_starts) > 1:
            holding_times = [
                (end - start).days
                for start, end in zip(trade_starts[:-1], trade_starts[1:])
            ]
            avg_hold_time = sum(holding_times) / len(
                holding_times) if holding_times else 0
        else:
            avg_hold_time = (df.index[-1] - df.index[0]).days

        #Calculate Maximum Drawdown
        peak = portfolio.cummax()
        drawdown = (portfolio - peak) / peak
        max_drawdown = drawdown.min() * 100

        return total_trades, winning_trades / len(
            returns) * 100, avg_hold_time, max_drawdown

    # Calculate metrics for each strategy
    ma_trades, ma_win_rate, ma_hold_time, ma_max_drawdown = calculate_strategy_metrics(
        df['MA_Signal'], df['MA_Returns'], df['MA_Portfolio'])
    rsi_trades, rsi_win_rate, rsi_hold_time, rsi_max_drawdown = calculate_strategy_metrics(
        df['RSI_Signal'], df['RSI_Returns'], df['RSI_Portfolio'])
    combined_trades, combined_win_rate, combined_hold_time, combined_max_drawdown = calculate_strategy_metrics(
        df['Combined_Signal'], df['Combined_Returns'],
        df['Combined_Portfolio'])
    bb_trades, bb_win_rate, bb_hold_time, bb_max_drawdown = calculate_strategy_metrics(
        df['BB_Signal'], df['BB_Returns'], df['BB_Portfolio'])
    momentum_trades, momentum_win_rate, momentum_hold_time, momentum_max_drawdown = calculate_strategy_metrics(
        df['Momentum_Signal'], df['Momentum_Returns'],
        df['Momentum_Portfolio'])
    triple_ma_trades, triple_ma_win_rate, triple_ma_hold_time, triple_ma_max_drawdown = calculate_strategy_metrics(
        df['Triple_MA_Signal'], df['Triple_MA_Returns'],
        df['Triple_MA_Portfolio'])
    abs_cash_trades, abs_cash_win_rate, abs_cash_hold_time, abs_cash_max_drawdown = calculate_strategy_metrics(
        df['Abs_Mom_Cash_Signal'], df['Abs_Mom_Cash_Returns'],
        df['Abs_Mom_Cash_Portfolio'])
    abs_bond_trades, abs_bond_win_rate, abs_bond_hold_time, abs_bond_max_drawdown = calculate_strategy_metrics(
        df['Abs_Mom_Bond_Signal'], df['Abs_Mom_Bond_Returns'],
        df['Abs_Mom_Bond_Portfolio'])
    dual_mom_trades, dual_mom_win_rate, dual_mom_hold_time, dual_mom_max_drawdown = calculate_strategy_metrics(
        df['Dual_Mom_Signal'], df['Dual_Mom_Returns'],
        df['Dual_Mom_Portfolio'])

    # Plot individual strategy performances
    strategies = {
        'Buy & Hold': 'Buy_Hold_Value',
        'Moving Average Crossover': 'MA_Portfolio',
        'RSI': 'RSI_Portfolio',
        'Combined MA+RSI': 'Combined_Portfolio',
        'Bollinger Bands': 'BB_Portfolio',
        'Momentum (RSI+MACD)': 'Momentum_Portfolio',
        'Triple MA': 'Triple_MA_Portfolio',
        '12M Absolute Momentum (Cash)': 'Abs_Mom_Cash_Portfolio',
        '12M Absolute Momentum (Bond)': 'Abs_Mom_Bond_Portfolio',
        'Dual Momentum (1.25x)': 'Dual_Mom_Portfolio'
    }

    colors = [
        'blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'gray',
        'cyan'
    ]

    # Individual strategy plots with buy/sell signals
    signal_columns = {
        'Buy & Hold': 'Buy_Hold_Signal',
        'Moving Average Crossover': 'MA_Signal',
        'RSI': 'RSI_Signal',
        'Combined MA+RSI': 'Combined_Signal',
        'Bollinger Bands': 'BB_Signal',
        'Momentum (RSI+MACD)': 'Momentum_Signal',
        'Triple MA': 'Triple_MA_Signal',
        '12M Absolute Momentum (Cash)': 'Abs_Mom_Cash_Signal',
        '12M Absolute Momentum (Bond)': 'Abs_Mom_Bond_Signal',
        'Dual Momentum (1.25x)': 'Dual_Mom_Signal'
    }

    for (name, column), color in zip(strategies.items(), colors):
        if name in ['RSI', 'Momentum (RSI+MACD)']:
            # Single graph for RSI and Momentum strategies
            fig, ax = plt.subplots(figsize=(15, 8))
            ax.plot(df.index, df[column], label=name, color=color)
            ax.set_title(f'{name} Strategy Performance')
            ax.set_ylabel('Portfolio Value')
            format_y_axis(ax)
            y_min, y_max = get_axis_limits(df[column], initial_capital)
            ax.set_ylim(y_min, y_max)
            ax.axhline(y=initial_capital,
                      color='gray',
                      linestyle='--',
                      alpha=0.5,
                      label='Initial Investment')
            ax.legend(loc='upper left')
            ax.grid(True)
        else:
            # Dual graph for other strategies
            fig, (ax1, ax2) = plt.subplots(2,
                                         1,
                                         figsize=(15, 10),
                                         height_ratios=[2, 1])

            # Plot portfolio value
            ax1.plot(df.index, df[column], label=name, color=color)
            ax1.set_title(f'{name} Strategy Performance')
            ax1.set_ylabel('Portfolio Value')
            format_y_axis(ax1)
            y_min, y_max = get_axis_limits(df[column], initial_capital)
            ax1.set_ylim(y_min, y_max)
            ax1.axhline(y=initial_capital,
                      color='gray',
                      linestyle='--',
                      alpha=0.5,
                      label='Initial Investment')
            ax1.legend(loc='upper left')
            ax1.grid(True)

            # Plot price and signals
            ax2.plot(df.index,
                    df['Close'],
                    color='gray',
                    alpha=0.6,
                    label='S&P 500')

            if signal_columns[name] in df.columns:
                if name == 'Buy & Hold':
                    # For Buy & Hold, only show one buy signal at the start
                    ax2.scatter([df.index[0]], [df['Close'].iloc[0]],
                                color='green',
                                marker='^',
                                s=100,
                                label='Buy Signal')
                elif 'Absolute Momentum' in name:
                    # For absolute momentum, show transitions between market and cash/bonds
                    enter_signals = df[df[signal_columns[name]].diff() == 1].index  # Enter market
                    exit_signals = df[df[signal_columns[name]].diff() == -1].index  # Exit to cash/bonds

                    ax2.scatter(enter_signals,
                                df.loc[enter_signals, 'Close'],
                                color='green',
                                marker='^',
                                s=100,
                                label='Enter Market')
                    ax2.scatter(exit_signals,
                                df.loc[exit_signals, 'Close'],
                                color='yellow',
                                marker='v',
                                s=100,
                                label='Switch to Cash' if '(Cash)' in name else 'Switch to Bonds')
                else:
                    # For traditional strategies, show long/short signals except RSI and Momentum
                    if name not in ['RSI', 'Momentum (RSI+MACD)']:
                        buy_signals = df[df[signal_columns[name]] == 1].index
                        sell_signals = df[df[signal_columns[name]] == -1].index

                        ax2.scatter(buy_signals,
                                    df.loc[buy_signals, 'Close'],
                                    color='green',
                                    marker='^',
                                    s=100,
                                    label='Buy Signal')
                        ax2.scatter(sell_signals,
                                    df.loc[sell_signals, 'Close'],
                                    color='red',
                                    marker='v',
                                    s=100,
                                    label='Sell Signal')

            ax2.set_title('Buy/Sell Signals')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('S&P 500 Price')
            ax2.legend(loc='upper left')
            ax2.grid(True)

            plt.tight_layout()
            plt.savefig(
                f'{save_dir}/{name.lower().replace(" ", "_")}_performance.png')
            plt.close()

    # Combined comparison plot
    plt.figure(figsize=(15, 8))
    ax = plt.gca()

    # Ensure we have enough colors for all strategies
    colors = plt.cm.tab20(np.linspace(0, 1, len(strategies)))

    for (name, column), color in zip(strategies.items(), colors):
        plt.plot(df.index, df[column], label=name, color=color, alpha=0.8)

    plt.title('Strategy Comparison')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    format_y_axis(ax)

    # Get combined min/max from all strategies
    all_values = pd.concat([df[col] for col in strategies.values()])
    y_min, y_max = get_axis_limits(all_values, initial_capital)
    plt.ylim(y_min, y_max)

    # Create strategy performance map
    strategy_metrics = {
        'Buy & Hold': {
            'Return':
            df['Buy_Hold_Value'].iloc[-1] / initial_capital * 100 - 100,
            'Total_Trades':
            buy_hold_trades,
            'Win_Rate': (buy_hold_winning_trades / len(df) * 100),
            'Avg_Hold_Time':
            buy_hold_avg_hold_time,
            'Max_Drawdown':
            calculate_strategy_metrics(
                pd.Series(1, index=df.index),  # Always invested
                df['Buy_Hold_Returns'],
                df['Buy_Hold_Value'])[
                    3]  # Get the max_drawdown from the tuple returned
        },
        'Moving Average Crossover': {
            'Return':
            df['MA_Portfolio'].iloc[-1] / initial_capital * 100 - 100,
            'Total_Trades': ma_trades,
            'Win_Rate': ma_win_rate,
            'Avg_Hold_Time': ma_hold_time,
            'Max_Drawdown': ma_max_drawdown
        },
        'RSI': {
            'Return':
            df['RSI_Portfolio'].iloc[-1] / initial_capital * 100 - 100,
            'Total_Trades': rsi_trades,
            'Win_Rate': rsi_win_rate,
            'Avg_Hold_Time': rsi_hold_time,
            'Max_Drawdown': rsi_max_drawdown
        },
        'Combined MA+RSI': {
            'Return':
            df['Combined_Portfolio'].iloc[-1] / initial_capital * 100 - 100,
            'Total_Trades': combined_trades,
            'Win_Rate': combined_win_rate,
            'Avg_Hold_Time': combined_hold_time,
            'Max_Drawdown': combined_max_drawdown
        },
        'Bollinger Bands': {
            'Return':
            df['BB_Portfolio'].iloc[-1] / initial_capital * 100 - 100,
            'Total_Trades': bb_trades,
            'Win_Rate': bb_win_rate,
            'Avg_Hold_Time': bb_hold_time,
            'Max_Drawdown': bb_max_drawdown
        },
        'Momentum (RSI+MACD)': {
            'Return':
            df['Momentum_Portfolio'].iloc[-1] / initial_capital * 100 - 100,
            'Total_Trades': momentum_trades,
            'Win_Rate': momentum_win_rate,
            'Avg_Hold_Time': momentum_hold_time,
            'Max_Drawdown': momentum_max_drawdown
        },
        'Triple MA': {
            'Return':
            df['Triple_MA_Portfolio'].iloc[-1] / initial_capital * 100 - 100,
            'Total_Trades': triple_ma_trades,
            'Win_Rate': triple_ma_win_rate,
            'Avg_Hold_Time': triple_ma_hold_time,
            'Max_Drawdown': triple_ma_max_drawdown
        },
        '12M Absolute Momentum (Cash)': {
            'Return':
            df['Abs_Mom_Cash_Portfolio'].iloc[-1] / initial_capital * 100 -
            100,
            'Total_Trades':
            abs_cash_trades,
            'Win_Rate':
            abs_cash_win_rate,
            'Avg_Hold_Time':
            abs_cash_hold_time,
            'Max_Drawdown':
            abs_cash_max_drawdown
        },
        '12M Absolute Momentum (Bond)': {
            'Return':
            df['Abs_Mom_Bond_Portfolio'].iloc[-1] / initial_capital * 100 -
            100,
            'Total_Trades':
            abs_bond_trades,
            'Win_Rate':
            abs_bond_win_rate,
            'Avg_Hold_Time':
            abs_bond_hold_time,
            'Max_Drawdown':
            abs_bond_max_drawdown
        },
        'Dual Momentum (1.25x)': {
            'Return':
            df['Dual_Mom_Portfolio'].iloc[-1] / initial_capital * 100 - 100,
            'Total_Trades': dual_mom_trades,
            'Win_Rate': dual_mom_win_rate,
            'Avg_Hold_Time': dual_mom_hold_time,
            'Max_Drawdown': dual_mom_max_drawdown
        }
    }

    # Calculate CAGR for each strategy
    years = (df.index[-1] - df.index[0]).days / 365.25
    for metrics in strategy_metrics.values():
        metrics['CAGR'] = ((
            (1 + metrics['Return'] / 100)**(1 / years)) - 1) * 100

    # Create legend labels with actual metrics
    legend_labels = [
        f'{name} (Return: {metrics["Return"]:.2f}%, CAGR: {metrics["CAGR"]:.2f}%)'
        for name, metrics in strategy_metrics.items()
    ]

    plt.legend(legend_labels, loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/all_strategies_comparison.png',
                bbox_inches='tight')
    plt.close()

    # Create bar chart comparing strategies
    plt.figure(figsize=(15, 8))
    returns = [metrics['Return'] for metrics in strategy_metrics.values()]
    plt.bar(strategy_metrics.keys(), returns, color=colors)
    plt.xticks(rotation=45, ha='right')
    timeframe = f"Strategy Returns Comparison ({df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')})"
    plt.title(timeframe)
    plt.ylabel('Total Return (%)')
    plt.grid(True, axis='y', alpha=0.3)

    # Add value labels on top of each bar
    for i, v in enumerate(returns):
        plt.text(i, v, f'{v:.1f}%', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(f'{save_dir}/strategy_returns_comparison.png',
                bbox_inches='tight')
    plt.close()

    # Calculate final values for return
    portfolio_results = {
        strategy: df[column].iloc[-1]
        for strategy, column in strategies.items()
    }
    print("\n" + "=" * 80)
    print("TRADING STRATEGY PERFORMANCE")
    print("=" * 80)
    print("\nStrategy Results:")
    for name, metrics in strategy_metrics.items():
        print(f"\n{name}:")
        print(f"Total Return: {metrics['Return']:.2f}%")
        print(f"Average Yearly Return (CAGR): {metrics['CAGR']:.2f}%")
        print(f"Final Value: ${portfolio_results[name]:,.2f}")
        print(f"Maximum Drawdown: {metrics['Max_Drawdown']:.2f}%")
        if 'Total_Trades' in metrics or name in [
                'Abs_Mom_Cash', 'Abs_Mom_Bond'
        ]:
            trades = metrics.get('Total_Trades', 0)
            win_rate = metrics.get('Win_Rate', 0)
            hold_time = metrics.get('Avg_Hold_Time', 0)
            print(f"Total Trades: {trades}")
            print(f"Win Rate: {win_rate:.2f}%")
            print(f"Average Hold Time: {hold_time:.1f} days")

    return portfolio_results