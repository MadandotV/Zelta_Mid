# Import necessary libraries
import backtrader as bt
import pandas as pd
import os
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

# Define Mean Reversion Strategy
class MeanReversionStrategy(bt.Strategy):
    params = (('period', 20), ('devfactor', 2),)

    def __init__(self):
        # Closing price from the data feed
        self.dataclose = self.datas[0].close

        # Simple Moving Average (SMA)
        self.sma = bt.indicators.SimpleMovingAverage(self.datas[0], period=self.params.period)

        # Bollinger Bands Indicator
        self.bollinger = bt.indicators.BollingerBands(self.datas[0], period=self.params.period, devfactor=self.params.devfactor)

    def next(self):
        # Check if we are in the market (if we have an open position)
        if not self.position:
            # If the close price is below the lower Bollinger band, we consider it oversold -> Buy signal
            if self.dataclose[0] < self.bollinger.lines.bot:
                self.buy()  # Execute buy order
        else:
            # If the close price is above the upper Bollinger band, we consider it overbought -> Sell signal
            if self.dataclose[0] > self.bollinger.lines.top:
                self.sell()  # Execute sell order


# Function to load data from CSV
def load_data(file_path):
    # Load the data into a DataFrame, ensuring datetime column is properly parsed
    data = pd.read_csv(file_path, parse_dates=['datetime'])

    # Print the first few rows for verification
    print(f"Loading data from: {file_path}")
    print(data.head())

    # Rename columns to match the format expected by Backtrader
    data.rename(columns={
        'datetime': 'Date',
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    }, inplace=True)

    # Ensure 'Date' column is in datetime format and set it as the index
    data.set_index('Date', inplace=True)

    # Return only the relevant columns
    return data[['Open', 'High', 'Low', 'Close', 'Volume']]


# Function to run the backtest
def run_backtest(data, cash=100000, commission=0.001):
    # Initialize backtrader's cerebro engine
    cerebro = bt.Cerebro()

    # Add data feed to cerebro
    data_feed = bt.feeds.PandasData(dataname=data)
    cerebro.adddata(data_feed)

    # Add the Mean Reversion strategy
    cerebro.addstrategy(MeanReversionStrategy)

    # Set initial cash and commission for the broker
    cerebro.broker.set_cash(cash)
    cerebro.broker.setcommission(commission=commission)

    # Add performance analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')

    # Print starting portfolio value
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

    # Run the backtest
    results = cerebro.run()

    # Print ending portfolio value
    print('Ending Portfolio Value: %.2f' % cerebro.broker.getvalue())

    # Extract analyzers' results
    sharpe = results[0].analyzers.sharpe.get_analysis()
    drawdown = results[0].analyzers.drawdown.get_analysis()

    # Print Sharpe Ratio and maximum drawdown
    print(f"Sharpe Ratio: {sharpe['sharperatio']}")
    print(f"Max Drawdown: {drawdown['max']['drawdown']}%")

    # Plot the results
    cerebro.plot()


# Set folder path to your Google Drive folder
folder_path = '/content/drive/MyDrive/Quant Guild/mid_zelta/data/'

# Load the different time frame datasets using the correct file format
data_1h = load_data(os.path.join(folder_path, 'btcusdt_1h.csv'))
data_3m = load_data(os.path.join(folder_path, 'btcusdt_3m.csv'))
data_5m = load_data(os.path.join(folder_path, 'btcusdt_5m.csv'))
data_15m = load_data(os.path.join(folder_path, 'btcusdt_15m.csv'))
data_30m = load_data(os.path.join(folder_path, 'btcusdt_30m.csv'))

# Run the backtest for one of the datasets (you can try with different time frames)
run_backtest(data_15m)  # Backtesting on 15-minute data

