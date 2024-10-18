import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np

class MissingValueFiller:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

    def load_data(self):
        """
        Load data from the CSV file.
        """
        self.data = pd.read_csv(self.file_path, parse_dates=['datetime'], index_col='datetime')
        print(f"Data loaded: {len(self.data)} rows.")

    def fill_missing_values(self):
        """
        Fill missing values using the mean of adjacent timeframes (window=2).
        """
        imputer = SimpleImputer(strategy='mean')

        # Loop through OHLCV columns
        for column in ['open', 'high', 'low', 'close', 'volume']:
            # Compute rolling mean for adjacent timeframes
            rolling_mean = self.data[column].rolling(window=2, min_periods=1).mean()

            # Replace missing values with rolling mean
            self.data[column] = np.where(self.data[column].isna(), rolling_mean, self.data[column])

        print("Missing values filled with the mean of adjacent timeframes.")

    def save_data(self, output_file):
        """
        Save the updated data with filled missing values to a new CSV file.
        """
        self.data.to_csv(output_file)
        print(f"Data saved to {output_file}")

# Example usage
if __name__ == "__main__":
    filler = MissingValueFiller(file_path="D:/QuantStuff/btcusdt_3m_heikin_ashi.csv")
    filler.load_data()
    filler.fill_missing_values()

    # Save the filled data to a new file
    filler.save_data("D:/QuantStuff/btcusdt_3m_heikin_ashi_filled.csv")
