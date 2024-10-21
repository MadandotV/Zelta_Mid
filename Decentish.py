import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

class EnhancedMFI_RSI_MACD_Stochastic_Strategy:
    def __init__(self, data, position_size=1000):
        self.data = data
        self.signals = pd.DataFrame(index=self.data.index)
        self.signals['signal'] = 0
        self.signals['position'] = 0
        self.position_size = position_size

    def calculate_mfi(self, period=14):
        typical_price = (self.data['high'] + self.data['low'] + self.data['close']) / 3
        money_flow = typical_price * self.data['volume']
        self.data['positive_flow'] = money_flow.where(typical_price.diff() > 0, 0)
        self.data['negative_flow'] = money_flow.where(typical_price.diff() < 0, 0)
        rolling_positive_flow = self.data['positive_flow'].rolling(window=period).sum()
        rolling_negative_flow = self.data['negative_flow'].rolling(window=period).sum()
        self.data['mfi'] = 100 - (100 / (1 + (rolling_positive_flow / rolling_negative_flow)))

    def calculate_rsi(self, period=14):
        delta = self.data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        self.data['rsi'] = 100 - (100 / (1 + rs))

    def calculate_macd(self):
        self.data['ema12'] = self.data['close'].ewm(span=12, adjust=False).mean()
        self.data['ema26'] = self.data['close'].ewm(span=26, adjust=False).mean()
        self.data['macd'] = self.data['ema12'] - self.data['ema26']
        self.data['signal_line'] = self.data['macd'].ewm(span=9, adjust=False).mean()

    def calculate_stochastic(self, period=14):
        l14 = self.data['low'].rolling(window=period).min()
        h14 = self.data['high'].rolling(window=period).max()
        self.data['stoch_k'] = 100 * ((self.data['close'] - l14) / (h14 - l14))
        self.data['stoch_d'] = self.data['stoch_k'].rolling(window=3).mean()

    def detect_divergence(self, price, indicator, window=14):
        price_change = price.diff(window)
        indicator_change = indicator.diff(window)
        bullish_div = (price_change < 0) & (indicator_change > 0)
        bearish_div = (price_change > 0) & (indicator_change < 0)
        return bullish_div, bearish_div

    def detect_crossover(self, series1, series2):
        crossover_up = (series1.shift(1) < series2.shift(1)) & (series1 > series2)
        crossover_down = (series1.shift(1) > series2.shift(1)) & (series1 < series2)
        return crossover_up, crossover_down

    def calculate_volatility(self, window=20):
        high_low = self.data['high'] - self.data['low']
        high_close = np.abs(self.data['high'] - self.data['close'].shift())
        low_close = np.abs(self.data['low'] - self.data['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        return true_range.rolling(window=window).mean()

    def detect_market_regime(self, window=20):
        returns = self.data['close'].pct_change()
        volatility = returns.rolling(window=window).std()
        trend = self.data['close'].rolling(window=window).mean()
        
        features = pd.concat([returns, volatility, trend], axis=1)
        features = features.dropna()
        
        kmeans = KMeans(n_clusters=3, random_state=42)
        features['regime'] = kmeans.fit_predict(features)
        
        self.data['market_regime'] = features['regime']
        self.data['market_regime'] = self.data['market_regime'].ffill()

    def calculate_position_size(self, signal_strength):
        return self.position_size * abs(signal_strength)

    def generate_signals(self):
        self.calculate_mfi()
        self.calculate_rsi()
        self.calculate_macd()
        self.calculate_stochastic()
        
        volatility = self.calculate_volatility()
        volatility_factor = volatility / volatility.mean()
        
        self.detect_market_regime()

        rsi_bull_div, rsi_bear_div = self.detect_divergence(self.data['close'], self.data['rsi'])
        macd_bull_div, macd_bear_div = self.detect_divergence(self.data['close'], self.data['macd'])
        macd_crossover_up, macd_crossover_down = self.detect_crossover(self.data['macd'], self.data['signal_line'])
        stoch_crossover_up, stoch_crossover_down = self.detect_crossover(self.data['stoch_k'], self.data['stoch_d'])

        for i in range(1, len(self.data)):
            vf = volatility_factor.iloc[i-1]
            regime = self.data['market_regime'].iloc[i-1]
            
            # Dynamic thresholds based on volatility and market regime
            if regime == 0:  # Ranging market
                mfi_low, mfi_high = 40 * vf, 60 / vf
                rsi_low, rsi_high = 40 * vf, 60 / vf
                stoch_low, stoch_high = 30 * vf, 70 / vf
            elif regime == 1:  # Trending market
                mfi_low, mfi_high = 30 * vf, 70 / vf
                rsi_low, rsi_high = 30 * vf, 70 / vf
                stoch_low, stoch_high = 20 * vf, 80 / vf
            else:  # Volatile market
                mfi_low, mfi_high = 45 * vf, 55 / vf
                rsi_low, rsi_high = 45 * vf, 55 / vf
                stoch_low, stoch_high = 40 * vf, 60 / vf

            # Generate signals
            if (self.data['mfi'].iloc[i-1] < mfi_low) and (self.data['rsi'].iloc[i-1] < rsi_low) and \
               macd_crossover_up.iloc[i-1] and (self.data['stoch_k'].iloc[i-1] < stoch_low) and \
               (rsi_bull_div.iloc[i-1] or macd_bull_div.iloc[i-1]):
                self.signals.loc[self.signals.index[i], 'signal'] = 1
            elif (self.data['mfi'].iloc[i-1] > mfi_high) and (self.data['rsi'].iloc[i-1] > rsi_high) and \
                 macd_crossover_down.iloc[i-1] and (self.data['stoch_k'].iloc[i-1] > stoch_high) and \
                 (rsi_bear_div.iloc[i-1] or macd_bear_div.iloc[i-1]):
                self.signals.loc[self.signals.index[i], 'signal'] = -1
            elif (self.data['mfi'].iloc[i-1] < mfi_low + 5) and (self.data['rsi'].iloc[i-1] < rsi_low + 5) and \
                 macd_crossover_up.iloc[i-1] and stoch_crossover_up.iloc[i-1]:
                self.signals.loc[self.signals.index[i], 'signal'] = 1
            elif (self.data['mfi'].iloc[i-1] > mfi_high - 5) and (self.data['rsi'].iloc[i-1] > rsi_high - 5) and \
                 macd_crossover_down.iloc[i-1] and stoch_crossover_down.iloc[i-1]:
                self.signals.loc[self.signals.index[i], 'signal'] = -1

        # Forward fill signals
        self.signals['signal'] = self.signals['signal'].ffill().fillna(0)

    def save_signals(self, output_file):
        output_data = self.data[['open', 'high', 'low', 'close', 'volume']].copy()
        output_data['signal'] = self.signals['signal']
        output_data.to_csv(output_file)

if __name__ == "__main__":
    # Load data
    data_path = 'D:/QuantStuff/btcusdt_3m_heikin_ashi_filled.csv'
    data = pd.read_csv(data_path, index_col='datetime', parse_dates=True)

    # Initialize the enhanced strategy with loaded data
    strategy = EnhancedMFI_RSI_MACD_Stochastic_Strategy(data)

    # Generate signals based on the enhanced strategy
    strategy.generate_signals()

    # Save the output with simplified signals (-1, 0, 1)
    output_file = 'D:/QuantStuff/btcusdt_3m_with_signals.csv'
    strategy.save_signals(output_file)
