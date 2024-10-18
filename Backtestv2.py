import pandas as pd
import numpy as np

def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
    """Calculate the Sharpe Ratio."""
    excess_returns = np.array(returns) - risk_free_rate
    return excess_returns.mean() / excess_returns.std(ddof=1)

def benchmark_returns(signals, brokerage_cost=0.001):
    """Calculate benchmark returns based on buy-and-hold strategy using signals."""
    returns = []
    year_wise_returns = {}
    initial_investment = 1000  # You can adjust this value as needed
    position = 0  # Current position (0 = no position, 1 = holding a position)
    entry_price = 0  # Price at which we entered the position

    for i in range(len(signals)):
        current_date = signals.index[i]

        # Buy signal
        if signals['signal'].iloc[i] == 1 and position == 0:  # Enter position
            position = 1
            entry_price = signals['close'].iloc[i] * (1 + brokerage_cost)  # Adjusting for brokerage

        # Sell signal
        elif signals['signal'].iloc[i] == -1 and position == 1:  # Exit position
            exit_price = signals['close'].iloc[i] * (1 - brokerage_cost)  # Adjusting for brokerage
            returns.append((exit_price - entry_price) / entry_price)  # Record return
            position = 0  # Reset position

            # Yearly performance tracking
            year = current_date.year
            if year not in year_wise_returns:
                year_wise_returns[year] = 0
            year_wise_returns[year] += (exit_price - entry_price)

    # If still holding at the end of the data, sell at the last price
    if position == 1:
        exit_price = signals['close'].iloc[-1] * (1 - brokerage_cost)  # Adjusting for brokerage
        returns.append((exit_price - entry_price) / entry_price)  # Final return

    # Calculate cumulative return
    cumulative_return = (1 + pd.Series(returns)).prod() - 1
    total_return = initial_investment * (1 + cumulative_return)

    # Calculate Sharpe Ratio
    sharpe_ratio = calculate_sharpe_ratio(returns)

    print(f"Total Portfolio Value after Benchmarking: ${total_return:.2f}")
    print(f"Cumulative Return: {cumulative_return:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

    print("\nYear-wise Returns:")
    for year, profit in year_wise_returns.items():
        print(f"{year}: ${profit:.2f}")

if __name__ == "__main__":
    # Load signals for benchmarking
    signals_path = 'D:/QuantStuff/btcusdt_3m_with_signals.csv'
    signals = pd.read_csv(signals_path, index_col='datetime', parse_dates=True)

    # Run the benchmarking function
    benchmark_returns(signals)
