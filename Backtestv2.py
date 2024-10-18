import pandas as pd
import numpy as np
from scipy import stats

def calculate_sharpe_ratio(returns, risk_free_rate=0.07):
    """Calculate the Sharpe Ratio."""
    excess_returns = np.array(returns) - risk_free_rate
    return excess_returns.mean() / excess_returns.std(ddof=1)


def calculate_sortino_ratio(returns, risk_free_rate=0.07, target_return=0):
    """Calculate the Sortino Ratio."""
    excess_returns = np.array(returns) - risk_free_rate
    downside_returns = excess_returns[excess_returns < target_return]
    if len(downside_returns) == 0:
        return np.inf  # No downside risk
    downside_std = np.std(downside_returns, ddof=1)
    if downside_std == 0:
        return np.inf  # To handle division by zero
    return (np.mean(excess_returns) - target_return) / downside_std

def calculate_max_drawdown(portfolio_values):
    """Calculate the maximum drawdown."""
    peak = portfolio_values[0]
    max_drawdown = 0
    for value in portfolio_values:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak
        if drawdown > max_drawdown:
            max_drawdown = drawdown
    return max_drawdown

def benchmark_returns(signals, initial_investment=1000, brokerage_cost=0.001, risk_free_rate=0.07):
    """Calculate benchmark returns based on signal changes with improved accuracy and additional metrics."""
    portfolio_value = initial_investment
    portfolio_values = [initial_investment]
    year_wise_returns = {}
    position = 0
    entry_price = 0
    entry_time = None

    total_trades = 0
    winning_trades = 0
    losing_trades = 0
    profit_trades = []
    loss_trades = []
    holding_durations = []
    trade_returns = []

    largest_winning_trade = 0
    largest_losing_trade = 0
    gross_profit = 0
    gross_loss = 0

    for i in range(1, len(signals)):
        current_signal = signals['signal'].iloc[i]
        prev_signal = signals['signal'].iloc[i - 1]
        current_date = signals.index[i]
        current_price = signals['close'].iloc[i]

        # Handle position changes
        if current_signal != prev_signal or (current_signal == 0 and position != 0):
            # Close existing position if any
            if position != 0:
                exit_price = current_price * (1 - brokerage_cost)
                trade_return = (exit_price / entry_price - 1) if position == 1 else (1 - exit_price / entry_price)
                portfolio_value *= (1 + trade_return)
                portfolio_values.append(portfolio_value)
                total_trades += 1
                trade_returns.append(trade_return)

                # Calculate holding duration
                holding_duration = (current_date - entry_time).total_seconds() / 3600  # in hours
                holding_durations.append(holding_duration)

                # Classify trade
                if trade_return > 0:
                    winning_trades += 1
                    profit_trades.append(trade_return)
                    gross_profit += trade_return * portfolio_value
                    largest_winning_trade = max(largest_winning_trade, trade_return)
                else:
                    losing_trades += 1
                    loss_trades.append(trade_return)
                    gross_loss += abs(trade_return * portfolio_value)
                    largest_losing_trade = min(largest_losing_trade, trade_return)

                # Yearly performance tracking
                year = current_date.year
                if year not in year_wise_returns:
                    year_wise_returns[year] = 1
                year_wise_returns[year] *= (1 + trade_return)

            # Open new position if signal is non-zero
            if current_signal != 0:
                entry_price = current_price * (1 + brokerage_cost)
                position = current_signal
                entry_time = current_date
            else:
                position = 0

        else:
            # Update portfolio value for the current price
            portfolio_values.append(portfolio_value * (current_price / signals['close'].iloc[i-1]))

    # Close any remaining position at the end
    if position != 0:
        exit_price = signals['close'].iloc[-1] * (1 - brokerage_cost)
        trade_return = (exit_price / entry_price - 1) if position == 1 else (1 - exit_price / entry_price)
        portfolio_value *= (1 + trade_return)
        portfolio_values.append(portfolio_value)
        total_trades += 1
        trade_returns.append(trade_return)

        holding_duration = (signals.index[-1] - entry_time).total_seconds() / 3600  # in hours
        holding_durations.append(holding_duration)

        if trade_return > 0:
            winning_trades += 1
            profit_trades.append(trade_return)
            gross_profit += trade_return * portfolio_value
            largest_winning_trade = max(largest_winning_trade, trade_return)
        else:
            losing_trades += 1
            loss_trades.append(trade_return)
            gross_loss += abs(trade_return * portfolio_value)
            largest_losing_trade = min(largest_losing_trade, trade_return)

        year = signals.index[-1].year
        if year not in year_wise_returns:
            year_wise_returns[year] = 1
        year_wise_returns[year] *= (1 + trade_return)

    # Calculate overall return
    overall_return = (portfolio_value / initial_investment) - 1

    # Calculate Sharpe and Sortino Ratios
    sharpe_ratio = calculate_sharpe_ratio(trade_returns, risk_free_rate)
    sortino_ratio = calculate_sortino_ratio(trade_returns, risk_free_rate)

    # Calculate max drawdown
    max_drawdown = calculate_max_drawdown(portfolio_values)

    # Calculate average holding duration
    avg_holding_duration = sum(holding_durations) / len(holding_durations) if holding_durations else 0

    # Print results
    print(f"Total Portfolio Value: ${portfolio_value:.2f}")
    print(f"Overall Return: {overall_return:.2%}")
    print(f"Total Trades: {total_trades}")
    print(f"Winning Trades: {winning_trades}")
    print(f"Losing Trades: {losing_trades}")
    print(f"Win Rate: {winning_trades / total_trades:.2%}" if total_trades else "No trades")
    print(f"Average Profit per Winning Trade: {sum(profit_trades) / winning_trades:.2%}" if winning_trades else "No winning trades")
    print(f"Average Loss per Losing Trade: {sum(loss_trades) / losing_trades:.2%}" if losing_trades else "No losing trades")
    print(f"Largest Winning Trade: {largest_winning_trade:.2%}")
    print(f"Largest Losing Trade: {largest_losing_trade:.2%}")
    print(f"Gross Profit: ${gross_profit:.2f}")
    print(f"Gross Loss: ${gross_loss:.2f}")
    print(f"Profit Factor: {gross_profit / abs(gross_loss):.2f}" if gross_loss != 0 else "Infinite")
    print(f"Average Holding Duration: {avg_holding_duration:.2f} hours")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Sortino Ratio: {sortino_ratio:.2f}")
    print(f"Maximum Drawdown: {max_drawdown:.2%}")

    print("\nYear-wise Returns:")
    for year, year_return in year_wise_returns.items():
        print(f"{year}: {year_return - 1:.2%}")

    return (portfolio_value, overall_return, total_trades, winning_trades, losing_trades, 
            sharpe_ratio, sortino_ratio, max_drawdown, avg_holding_duration, 
            largest_winning_trade, largest_losing_trade, gross_profit, gross_loss, 
            year_wise_returns)
if __name__ == "__main__":
    # Load signals for benchmarking
    signals_path = 'D:/QuantStuff/btcusdt_3m_with_signals.csv'
    signals = pd.read_csv(signals_path, index_col='datetime', parse_dates=True)

    # Run the benchmarking function
    benchmark_returns(signals)
