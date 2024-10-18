import pandas as pd
import numpy as np

class MFI_RSI_MACD_Stochastic_Strategy:
    def __init__(self, data):
        self.data = data
        self.signals = pd.DataFrame(index=self.data.index)
        self.signals['signal'] = 0  # Initialize signal column
        self.signals['detailed_signal'] = 0.0  # Initialize detailed signal column as float    
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

    def detect_divergence(self, price, indicator, window=14):
        """Detect divergence between price and indicator."""
        price_change = price.diff(window)
        indicator_change = indicator.diff(window)
        
        bullish_div = (price_change < 0) & (indicator_change > 0)
        bearish_div = (price_change > 0) & (indicator_change < 0)
        
        return bullish_div, bearish_div

    def detect_crossover(self, series1, series2):
        """Detect crossover between two series."""
        crossover_up = (series1.shift(1) < series2.shift(1)) & (series1 > series2)
        crossover_down = (series1.shift(1) > series2.shift(1)) & (series1 < series2)
        return crossover_up, crossover_down

    def generate_signals(self):
        """Generate trading signals based on MFI, RSI, MACD, Stochastic Oscillator, divergence, and crossover."""
        self.calculate_mfi()
        self.calculate_rsi()
        self.calculate_macd()
        self.calculate_stochastic()

        # Detect divergences
        rsi_bull_div, rsi_bear_div = self.detect_divergence(self.data['close'], self.data['rsi'])
        macd_bull_div, macd_bear_div = self.detect_divergence(self.data['close'], self.data['macd'])

        # Detect crossovers
        macd_crossover_up, macd_crossover_down = self.detect_crossover(self.data['macd'], self.data['signal_line'])
        stoch_crossover_up, stoch_crossover_down = self.detect_crossover(self.data['stoch_k'], self.data['stoch_d'])

        for i in range(1, len(self.data)):
            # Generate detailed signals
            if (self.data['mfi'].iloc[i] < 30) and (self.data['rsi'].iloc[i] < 30) and \
               macd_crossover_up.iloc[i] and (self.data['stoch_k'].iloc[i] < 20) and \
               (rsi_bull_div.iloc[i] or macd_bull_div.iloc[i]):
                self.signals.loc[self.signals.index[i], 'detailed_signal'] = 1.0   # Strong Buy signal

            elif (self.data['mfi'].iloc[i] > 70) and (self.data['rsi'].iloc[i] > 70) and \
                 macd_crossover_down.iloc[i] and (self.data['stoch_k'].iloc[i] > 80) and \
                 (rsi_bear_div.iloc[i] or macd_bear_div.iloc[i]):
                self.signals.loc[self.signals.index[i], 'detailed_signal'] = -1.0  # Strong Sell signal

            elif (self.data['mfi'].iloc[i] < 40) and (self.data['rsi'].iloc[i] < 40) and \
                 macd_crossover_up.iloc[i] and stoch_crossover_up.iloc[i]:
                self.signals.loc[self.signals.index[i], 'detailed_signal'] = 0.5  # Weak Buy signal

            elif (self.data['mfi'].iloc[i] > 60) and (self.data['rsi'].iloc[i] > 60) and \
                 macd_crossover_down.iloc[i] and stoch_crossover_down.iloc[i]:
                self.signals.loc[self.signals.index[i], 'detailed_signal'] = -0.5  # Weak Sell signal

        # Forward fill detailed signals
        self.signals['detailed_signal'] = self.signals['detailed_signal'].ffill().fillna(0.0)

        # Generate simplified signals for CSV output
        self.signals['signal'] = np.where(self.signals['detailed_signal'] > 0, 1,
                                          np.where(self.signals['detailed_signal'] < 0, -1, 0))

    def save_signals(self, output_file):
        """Save the signals along with OHLCV data to a CSV file."""
        # Merge the signals with OHLCV data
        output_data = self.data[['open', 'high', 'low', 'close', 'volume']].copy()
        output_data['signal'] = self.signals['signal'].astype(int)
        
        # Save in the order of datetime, open, high, low, close, volume, signal
        output_data.to_csv(output_file)

    def get_detailed_signals(self):
        """Return the DataFrame with detailed signals."""
        return self.signals[['detailed_signal']]

if __name__ == "__main__":
    # Load data
    data_path = 'D:/QuantStuff/btcusdt_3m_heikin_ashi_filled.csv'
    data = pd.read_csv(data_path, index_col='datetime', parse_dates=True)

    # Initialize the enhanced strategy with loaded data
    strategy = MFI_RSI_MACD_Stochastic_Strategy(data)

    # Generate signals based on the enhanced strategy
    strategy.generate_signals()

    # Save the output with simplified signals (-1, 0, 1)
    output_file = 'D:/QuantStuff/btcusdt_3m_with_signals.csv'
    strategy.save_signals(output_file)
