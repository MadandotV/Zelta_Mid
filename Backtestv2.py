import pandas as pd
import numpy as np

def calculate_sharpe_ratio(returns, risk_free_rate=0.07):
    """Calculate the Sharpe Ratio."""
    excess_returns = np.array(returns) - risk_free_rate
    return excess_returns.mean() / excess_returns.std(ddof=1)

def benchmark_returns(signals, brokerage_cost=0.001):
    """Calculate benchmark returns based on signal changes."""
    returns = []
    year_wise_returns = {}
    initial_investment = 1000  # You can adjust this value as needed
    position = 0  # Current position (0 = no position, 1 = holding a buy position, -1 = holding a sell position)
    entry_price = 0  # Price at which we entered the position

    total_trades = 0
    winning_trades = 0
    losing_trades = 0
    total_profit = 0
    total_loss = 0
    profit_trades = []
    loss_trades = []

    # Iterate over the signals to evaluate changes in signal values
    for i in range(1, len(signals)):
        current_signal = signals['signal'].iloc[i]
        prev_signal = signals['signal'].iloc[i - 1]
        current_date = signals.index[i]

        # Handle transitions from 1 to -1 (or -1 to 1)
        if current_signal == -1 and prev_signal == 1 and position == 1:
            # Close the long position and open a short
            exit_price = signals['close'].iloc[i] * (1 - brokerage_cost)
            trade_return = (exit_price - entry_price) / entry_price
            returns.append(trade_return)
            total_trades += 1

            # Classify trade
            if trade_return > 0:
                winning_trades += 1
                total_profit += trade_return
                profit_trades.append(trade_return)
            else:
                losing_trades += 1
                total_loss += trade_return
                loss_trades.append(trade_return)

            # Enter new short position
            entry_price = signals['close'].iloc[i] * (1 + brokerage_cost)
            position = -1  # Update to short

        elif current_signal == 1 and prev_signal == -1 and position == -1:
            # Close the short position and open a long
            exit_price = signals['close'].iloc[i] * (1 - brokerage_cost)
            trade_return = (entry_price - exit_price) / entry_price  # Reverse for short position
            returns.append(trade_return)
            total_trades += 1

            # Classify trade
            if trade_return > 0:
                winning_trades += 1
                total_profit += trade_return
                profit_trades.append(trade_return)
            else:
                losing_trades += 1
                total_loss += trade_return
                loss_trades.append(trade_return)

            # Enter new long position
            entry_price = signals['close'].iloc[i] * (1 + brokerage_cost)
            position = 1  # Update to long

        # If signal changes from 0 to non-zero (enter position)
        elif current_signal != 0 and position == 0:
            entry_price = signals['close'].iloc[i] * (1 + brokerage_cost)
            position = current_signal  # Track if we are entering a buy (1) or sell (-1)

        # If signal changes from non-zero to 0 (exit position)
        elif current_signal == 0 and position != 0:
            exit_price = signals['close'].iloc[i] * (1 - brokerage_cost)
            trade_return = (exit_price - entry_price) / entry_price
            if position == -1:
                trade_return = -trade_return  # Reverse return calculation for short trades
            
            returns.append(trade_return)  # Record return
            total_trades += 1
            position = 0  # Reset position

            # Classify trade as profit or loss
            if trade_return > 0:
                winning_trades += 1
                total_profit += trade_return
                profit_trades.append(trade_return)
            else:
                losing_trades += 1
                total_loss += trade_return
                loss_trades.append(trade_return)

            # Yearly performance tracking
            year = current_date.year
            if year not in year_wise_returns:
                year_wise_returns[year] = 0
            year_wise_returns[year] += (exit_price - entry_price)

    # If still holding a position at the end of the data, exit at the last price
    if position != 0:
        exit_price = signals['close'].iloc[-1] * (1 - brokerage_cost)
        trade_return = (exit_price - entry_price) / entry_price
        if position == -1:
            trade_return = -trade_return  # Reverse return for short
        returns.append(trade_return)  # Final return
        total_trades += 1
        if trade_return > 0:
            winning_trades += 1
            total_profit += trade_return
            profit_trades.append(trade_return)
        else:
            losing_trades += 1
            total_loss += trade_return
            loss_trades.append(trade_return)

    # Calculate cumulative return
    cumulative_return = (1 + pd.Series(returns)).prod() - 1
    total_return = initial_investment * (1 + cumulative_return)

    # Calculate Sharpe Ratio
    sharpe_ratio = calculate_sharpe_ratio(returns)

    # Average profit/loss per trade
    avg_profit_per_trade = total_profit / winning_trades if winning_trades > 0 else 0
    avg_loss_per_trade = total_loss / losing_trades if losing_trades > 0 else 0

    # Output results
    print(f"Total Portfolio Value after Benchmarking: ${total_return:.2f}")
    print(f"Cumulative Return: {cumulative_return:.2%}")
    print(f"Total Trades: {total_trades}")
    print(f"Winning Trades: {winning_trades}")
    print(f"Losing Trades: {losing_trades}")
    print(f"Average Profit per Trade: {avg_profit_per_trade:.2%}")
    print(f"Average Loss per Trade: {avg_loss_per_trade:.2%}")
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
