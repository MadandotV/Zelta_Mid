import pandas as pd
import numpy as np

class HeikinAshiDenoiser:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.heikin_ashi_data = None

    def load_data(self):
        """
        Load BTC/USD data from a CSV file.
        """
        self.data = pd.read_csv(self.file_path, parse_dates=['datetime'], index_col='datetime')
        print(f"Data loaded: {len(self.data)} rows.")

    def calculate_heikin_ashi(self):
        """
        Calculate Heikin Ashi candles to denoise the price data.
        """
        self.heikin_ashi_data = pd.DataFrame(index=self.data.index)

        # Heikin Ashi close: (Open + High + Low + Close) / 4
        self.heikin_ashi_data['close'] = (self.data['open'] + self.data['high'] + self.data['low'] + self.data['close']) / 4

        # Initialize ha_open as a series filled with NaNs
        self.heikin_ashi_data['open'] = np.nan

        # Set the first value of ha_open to the average of open and close
        self.heikin_ashi_data.loc[self.heikin_ashi_data.index[0], 'open'] = (self.data['open'].iloc[0] + self.data['close'].iloc[0]) / 2

        # Fill the rest of ha_open using the previous ha_open and ha_close
        for i in range(1, len(self.heikin_ashi_data)):
            self.heikin_ashi_data.loc[self.heikin_ashi_data.index[i], 'open'] = (
                self.heikin_ashi_data.loc[self.heikin_ashi_data.index[i-1], 'open'] +
                self.heikin_ashi_data.loc[self.heikin_ashi_data.index[i-1], 'close']
            ) / 2

        # Heikin Ashi high: max(high, ha_open, ha_close)
        self.heikin_ashi_data['high'] = self.data[['high']].join(self.heikin_ashi_data[['open', 'close']]).max(axis=1)

        # Heikin Ashi low: min(low, ha_open, ha_close)
        self.heikin_ashi_data['low'] = self.data[['low']].join(self.heikin_ashi_data[['open', 'close']]).min(axis=1)

        # Heikin Ashi volume remains the same
        self.heikin_ashi_data['volume'] = self.data['volume']

        # Replace missing values with nearby values
        self.heikin_ashi_data = self.heikin_ashi_data.fillna(self.heikin_ashi_data.interpolate(method='nearest'))

    def save_heikin_ashi_data(self, output_file_path):
        """
        Save the Heikin Ashi data to a new CSV file with the desired column order.
        """
        # Define the desired column order
        column_order = ['open', 'high', 'low', 'close', 'volume']

        # Create a new DataFrame with the specified order
        ordered_data = self.heikin_ashi_data[column_order]
        
        # Add datetime as the index to the DataFrame
        ordered_data.index.name = 'datetime'

        # Save to CSV
        ordered_data.to_csv(output_file_path)
        print(f"Heikin Ashi data saved to {output_file_path}")

# Example usage
if __name__ == "__main__": 
    # Initialize the Heikin Ashi Denoiser
    denoiser = HeikinAshiDenoiser(file_path="D:/QuantStuff/btcusdt_3m.csv")
    
    # Load the data
    denoiser.load_data()
    
    # Calculate Heikin Ashi candles
    denoiser.calculate_heikin_ashi()

    # Save the denoised data
    output_file = "D:/QuantStuff/btcusdt_3m_heikin_ashi.csv"
    denoiser.save_heikin_ashi_data(output_file)
