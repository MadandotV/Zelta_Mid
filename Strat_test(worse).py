import pandas as pd
import numpy as np

class EnhancedMFI_RSI_MACD_Stochastic_Strategy:
    def __init__(self, data):
        self.data = data
        self.signals = pd.DataFrame(index=self.data.index)
        self.signals['signal'] = 0  # Initialize signal column
        self.signals['entry_price'] = 0.0  # Initialize entry price column
        self.signals['exit_price'] = 0.0  # Initialize exit price column
        self.signals['trailing_stop'] = 0.0  # Initialize trailing stop column

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
    def calculate_atr(self, period=14):
        """Calculate the Average True Range (ATR)."""
        high_low = self.data['high'] - self.data['low']
        high_close = np.abs(self.data['high'] - self.data['close'].shift())
        low_close = np.abs(self.data['low'] - self.data['close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        
        self.data['atr'] = true_range.rolling(window=period).mean()
    def calculate_moving_average(self, period=50):
        """Calculate the moving average."""
        self.data['ma'] = self.data['close'].rolling(window=period).mean()

    def generate_signals(self):
        """Generate trading signals with improved exit conditions."""
        self.calculate_mfi()
        self.calculate_rsi()
        self.calculate_macd()
        self.calculate_stochastic()
        self.calculate_atr()
        self.calculate_moving_average()

        # Detect divergences
        rsi_bull_div, rsi_bear_div = self.detect_divergence(self.data['close'], self.data['rsi'])
        macd_bull_div, macd_bear_div = self.detect_divergence(self.data['close'], self.data['macd'])

        # Detect crossovers
        macd_crossover_up, macd_crossover_down = self.detect_crossover(self.data['macd'], self.data['signal_line'])
        stoch_crossover_up, stoch_crossover_down = self.detect_crossover(self.data['stoch_k'], self.data['stoch_d'])

        current_signal = 0
        trailing_stop = 0
        entry_price = 0
        atr_multiple = 3  # Use 3 times ATR for take profit
        trailing_stop_multiple = 2  # Use 2 times ATR for trailing stop

        for i in range(1, len(self.data)):
            # Check for buy signal
            if ((self.data['mfi'].iloc[i] < 35) and (self.data['rsi'].iloc[i] < 35) and 
                macd_crossover_up.iloc[i] and (self.data['stoch_k'].iloc[i] < 25) and 
                (rsi_bull_div.iloc[i] or macd_bull_div.iloc[i])) or \
               ((self.data['mfi'].iloc[i] < 45) and (self.data['rsi'].iloc[i] < 45) and 
                macd_crossover_up.iloc[i] and stoch_crossover_up.iloc[i]):
                current_signal = 1  # Set buy signal

            # Check for sell signal
            elif ((self.data['mfi'].iloc[i] > 65) and (self.data['rsi'].iloc[i] > 65) and 
                  macd_crossover_down.iloc[i] and (self.data['stoch_k'].iloc[i] > 75) and 
                  (rsi_bear_div.iloc[i] or macd_bear_div.iloc[i])) or \
                 ((self.data['mfi'].iloc[i] > 55) and (self.data['rsi'].iloc[i] > 55) and 
                  macd_crossover_down.iloc[i] and stoch_crossover_down.iloc[i]):
                current_signal = -1  # Set sell signal

            # Exit conditions
            elif current_signal == 1:  # Long position
                # Update trailing stop
                trailing_stop = max(trailing_stop, self.data['close'].iloc[i] - (trailing_stop_multiple * self.data['atr'].iloc[i]))
                
                # Check exit conditions
                if (self.data['close'].iloc[i] <= trailing_stop or  # Trailing stop hit
                    self.data['close'].iloc[i] >= entry_price + (atr_multiple * self.data['atr'].iloc[i]) or  # Take profit hit
                    self.data['close'].iloc[i] < self.data['ma'].iloc[i]):  # Price crosses below MA
                    current_signal = 0
                    self.signals.at[self.signals.index[i], 'exit_price'] = self.data['close'].iloc[i]

            elif current_signal == -1:  # Short position
                # Update trailing stop
                trailing_stop = min(trailing_stop, self.data['close'].iloc[i] + (trailing_stop_multiple * self.data['atr'].iloc[i]))
                
                # Check exit conditions
                if (self.data['close'].iloc[i] >= trailing_stop or  # Trailing stop hit
                    self.data['close'].iloc[i] <= entry_price - (atr_multiple * self.data['atr'].iloc[i]) or  # Take profit hit
                    self.data['close'].iloc[i] > self.data['ma'].iloc[i]):  # Price crosses above MA
                    current_signal = 0
                    self.signals.at[self.signals.index[i], 'exit_price'] = self.data['close'].iloc[i]

            self.signals.at[self.signals.index[i], 'signal'] = current_signal
            self.signals.at[self.signals.index[i], 'entry_price'] = entry_price if current_signal != 0 else 0
            self.signals.at[self.signals.index[i], 'trailing_stop'] = trailing_stop if current_signal != 0 else 0

    def save_signals(self, output_file):
        """Save the signals along with OHLCV data to a CSV file."""
        output_data = self.data[['open', 'high', 'low', 'close', 'volume', 'atr', 'ma']].copy()
        output_data['signal'] = self.signals['signal']
        output_data['entry_price'] = self.signals['entry_price']
        output_data['exit_price'] = self.signals['exit_price']
        output_data['trailing_stop'] = self.signals['trailing_stop']
        
        output_data.to_csv(output_file)

if __name__ == "__main__":
    # Load data
    data_path = 'D:/QuantStuff/btcusdt_3m_heikin_ashi_filled.csv'
    data = pd.read_csv(data_path, index_col='datetime', parse_dates=True)

    # Initialize the enhanced strategy with loaded data
    strategy = EnhancedMFI_RSI_MACD_Stochastic_Strategy(data)

    # Generate signals based on the enhanced strategy
    strategy.generate_signals()

    # Save the output with signals (-1, 0, 1)
    output_file = 'D:/QuantStuff/btcusdt_3m_with_signals.csv'
    strategy.save_signals(output_file)