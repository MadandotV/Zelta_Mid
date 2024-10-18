import pandas as pd
import numpy as np

class MFI_RSI_MACD_Stochastic_Strategy:
    def __init__(self, data):
        self.data = data
        self.signals = pd.DataFrame(index=self.data.index)
        self.signals['signal'] = 0  # Initialize signal column

    def calculate_mfi(self, period=14):
        """Calculate the Money Flow Index (MFI)."""
        typical_price = (self.data['high'] + self.data['low'] + self.data['close']) / 3
        money_flow = typical_price * self.data['volume']
        self.data['positive_flow'] = money_flow.where(typical_price.diff() > 0, 0)
        self.data['negative_flow'] = money_flow.where(typical_price.diff() < 0, 0)

        # Calculate MFI
        rolling_positive_flow = self.data['positive_flow'].rolling(window=period).sum()
        rolling_negative_flow = self.data['negative_flow'].rolling(window=period).sum()
        self.data['mfi'] = 100 - (100 / (1 + (rolling_positive_flow / rolling_negative_flow)))

    def calculate_rsi(self, period=14):
        """Calculate the Relative Strength Index (RSI)."""
        delta = self.data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        self.data['rsi'] = 100 - (100 / (1 + rs))

    def calculate_macd(self):
        """Calculate the MACD."""
        self.data['ema12'] = self.data['close'].ewm(span=12, adjust=False).mean()
        self.data['ema26'] = self.data['close'].ewm(span=26, adjust=False).mean()
        self.data['macd'] = self.data['ema12'] - self.data['ema26']
        self.data['signal_line'] = self.data['macd'].ewm(span=9, adjust=False).mean()

    def calculate_stochastic(self, period=14):
        """Calculate the Stochastic Oscillator."""
        l14 = self.data['low'].rolling(window=period).min()
        h14 = self.data['high'].rolling(window=period).max()
        self.data['stoch_k'] = 100 * ((self.data['close'] - l14) / (h14 - l14))
        self.data['stoch_d'] = self.data['stoch_k'].rolling(window=3).mean()  # %D line

    def generate_signals(self):
        """Generate trading signals based on MFI, RSI, MACD, and Stochastic Oscillator."""
        self.calculate_mfi()
        self.calculate_rsi()
        self.calculate_macd()
        self.calculate_stochastic()

        for i in range(1, len(self.data)):
            # Conditions for generating signals
            if (self.data['mfi'].iloc[i] > 70) and (self.data['rsi'].iloc[i] > 70) and \
               (self.data['macd'].iloc[i] < self.data['signal_line'].iloc[i]) and \
               (self.data['stoch_k'].iloc[i] > 80):
                self.signals.loc[self.signals.index[i], 'signal'] = -1  # Sell signal

            elif (self.data['mfi'].iloc[i] < 30) and (self.data['rsi'].iloc[i] < 30) and \
                 (self.data['macd'].iloc[i] > self.data['signal_line'].iloc[i]) and \
                 (self.data['stoch_k'].iloc[i] < 20):
                self.signals.loc[self.signals.index[i], 'signal'] = 1   # Buy signal

        # Forward fill signals to maintain position
        self.signals['signal'] = self.signals['signal'].ffill()

    def save_signals(self, output_file):
        """Save the signals along with OHLCV data to a CSV file."""
        # Merge the signals with OHLCV data
        output_data = self.data[['open', 'high', 'low', 'close', 'volume']].copy()
        output_data['signal'] = self.signals['signal']
        
        # Save in the order of datetime, open, high, low, close, volume, signal
        output_data.to_csv(output_file)

if __name__ == "__main__":
    # Load data
    data_path = 'D:/QuantStuff/btcusdt_3m_heikin_ashi_filled.csv'
    data = pd.read_csv(data_path, index_col='datetime', parse_dates=True)

    # Initialize the strategy with loaded data
    strategy = MFI_RSI_MACD_Stochastic_Strategy(data)

    # Generate signals based on the strategy
    strategy.generate_signals()

    # Save the output with signals
    output_file = 'D:/QuantStuff/btcusdt_3m_with_signals.csv'
    strategy.save_signals(output_file)
