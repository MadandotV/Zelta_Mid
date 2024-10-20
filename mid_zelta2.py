import backtrader as bt
import pandas as pd
import numpy as np
import os
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

# Step 1: Define utility functions for KPIs
def calculate_sharpe_ratio(returns, risk_free_rate=0.07):
    excess_returns = np.array(returns) - risk_free_rate
    return excess_returns.mean() / excess_returns.std(ddof=1)

def calculate_sortino_ratio(returns, risk_free_rate=0.07, target_return=0):
    excess_returns = np.array(returns) - risk_free_rate
    downside_returns = excess_returns[excess_returns < target_return]
    if len(downside_returns) == 0:
        return np.inf  # No downside risk
    downside_std = np.std(downside_returns, ddof=1)
    if downside_std == 0:
        return np.inf  # Handle division by zero
    return (np.mean(excess_returns) - target_return) / downside_std

def calculate_max_drawdown(portfolio_values):
    peak = portfolio_values[0]
    max_drawdown = 0
    for value in portfolio_values:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak
        if drawdown > max_drawdown:
            max_drawdown = drawdown
    return max_drawdown

# Step 2: Define the enhanced strategy
class EnhancedStrategy(bt.Strategy):
    params = (('period', 20), ('devfactor', 2), ('rsi_lower', 30), ('rsi_upper', 70), ('atr_period', 14), ('trailing_stop_factor', 3))

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.sma = bt.indicators.SimpleMovingAverage(self.datas[0], period=self.params.period)
        self.bollinger = bt.indicators.BollingerBands(self.datas[0], period=self.params.period, devfactor=self.params.devfactor)
        self.rsi = bt.indicators.RelativeStrengthIndex(self.datas[0])
        self.atr = bt.indicators.ATR(self.datas[0], period=self.params.atr_period)

        # Track trades and metrics
        self.trades = []
        self.trade_returns = []
        self.initial_value = self.broker.getvalue()

    def next(self):
        # Entry condition (RSI and Bollinger Band-based)
        if not self.position:
            if self.dataclose[0] < self.bollinger.lines.bot and self.rsi < self.params.rsi_lower:
                self.buy()
                self.entry_price = self.dataclose[0]
                self.trailing_stop = self.entry_price - (self.params.trailing_stop_factor * self.atr[0])

        # Exit condition (RSI, Bollinger Bands, or trailing stop)
        else:
            if self.dataclose[0] > self.bollinger.lines.top or self.rsi > self.params.rsi_upper or self.dataclose[0] < self.trailing_stop:
                self.sell()
                trade_return = (self.dataclose[0] / self.entry_price - 1)
                self.trade_returns.append(trade_return)

    def stop(self):
        final_value = self.broker.getvalue()
        overall_return = (final_value / self.initial_value) - 1
        portfolio_values = [self.broker.getvalue() for _ in range(len(self))]

        # Calculate key performance metrics
        sharpe_ratio = calculate_sharpe_ratio(self.trade_returns)
        sortino_ratio = calculate_sortino_ratio(self.trade_returns)
        max_drawdown = calculate_max_drawdown(portfolio_values)

        print(f"Final Portfolio Value: {final_value:.2f}")
        print(f"Overall Return: {overall_return:.2%}")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"Sortino Ratio: {sortino_ratio:.2f}")
        print(f"Max Drawdown: {max_drawdown:.2%}")

# Step 3: Function to load data
def load_data(file_path):
    data = pd.read_csv(file_path, parse_dates=['datetime'])
    print(f"Loading data from: {file_path}")
    data.rename(columns={'datetime': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
    data.set_index('Date', inplace=True)
    return data[['Open', 'High', 'Low', 'Close', 'Volume']]

# Step 4: Backtest execution function with optimization
def run_backtest(data, cash=100000, commission=0.001):
    cerebro = bt.Cerebro()

    # Add data feed
    data_feed = bt.feeds.PandasData(dataname=data)
    cerebro.adddata(data_feed)

    # Add strategy with optimization
    cerebro.optstrategy(EnhancedStrategy,
                        period=range(10, 40, 5),
                        devfactor=range(1, 3),
                        rsi_lower=range(25, 35, 5),
                        rsi_upper=range(65, 75, 5),
                        atr_period=range(10, 20, 5),
                        trailing_stop_factor=range(2, 4))

    # Set broker cash and commission
    cerebro.broker.set_cash(cash)
    cerebro.broker.setcommission(commission=commission)

    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')

    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

    # Run backtest
    results = cerebro.run()

    print('Ending Portfolio Value: %.2f' % cerebro.broker.getvalue())

    sharpe = results[0].analyzers.sharpe.get_analysis()
    drawdown = results[0].analyzers.drawdown.get_analysis()

    print(f"Sharpe Ratio: {sharpe['sharperatio']}")
    print(f"Max Drawdown: {drawdown['max']['drawdown']}%")

    # Plot the results
    cerebro.plot()

# Step 5: Main execution
if __name__ == "__main__":
    folder_path = '/content/drive/MyDrive/Quant Guild/mid_zelta/data/'
    data_15m = load_data(os.path.join(folder_path, 'btcusdt_15m.csv'))

    # Run the backtest with optimization
    run_backtest(data_15m)

