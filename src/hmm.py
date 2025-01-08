#exploring the HMM model for algorithmic trading

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
import yfinance as yf
from datetime import datetime, timedelta
import pytz

print("Starting HMM SP500 analysis...")

# Get the current date and set start and end for today's trading day
today = datetime.now()
start_date = today.replace(hour=0, minute=0, second=0, microsecond=0)  # Beginning of today
end_date = today.replace(hour=23, minute=59, second=59, microsecond=999999)  # End of today

# Download S&P 500 data
sp500 = yf.download("^GSPC", start = start_date, end=  end_date, interval = "1m")

# Convert to Central Time (CST/CDT)
sp500.index = sp500.index.tz_convert('US/Central')

print("\n\n") # Add some space for better readability

print("Here Are The Latest S&P 500 Close Prices:")
print(sp500['Close', '^GSPC'].tail(30))

print("\n\n") # Add some space for better readability

# Calculate daily returns
sp500['Log_Returns'] = sp500['Close'].pct_change().apply(np.log)

# Drop NaN values in Log_Returns 
sp500.dropna(subset=[('Log_Returns', '')], inplace=True)

# Calculate rolling volatility (24hr window)
sp500['Volatility'] = sp500['Log_Returns'].rolling(window=15).std()

# Plot both volatility and price on the same figure
fig, ax1 = plt.subplots(figsize=(24, 12))

# Plot price on the first y-axis
ax1.set_xlabel("Date", fontsize=20)
ax1.set_ylabel("Price (Close)", color='tab:blue', fontsize=20)
ax1.plot(sp500.index, sp500['Close'], color='tab:blue', label="Price")
ax1.tick_params(axis='y', labelcolor='tab:blue', labelsize=16)

# Create a second y-axis to plot volatility
ax2 = ax1.twinx()  # this shares the same x-axis as ax1
ax2.set_ylabel("Volatility", color='tab:orange', fontsize=20)
ax2.plot(sp500.index, sp500['Volatility'], color='tab:orange', label="Volatility")
ax2.tick_params(axis='y', labelcolor='tab:orange', labelsize=16)

# Format the x-axis (for minute-level data)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax1.xaxis.set_major_locator(mdates.MinuteLocator(interval=30)) # Show every 30min
ax1.tick_params(axis='x', labelsize=16)  

# Add title and show plot
plt.title("S&P 500 1m Price and Volatility")
fig.tight_layout()  # adjusts the plot to ensure there's no overlap

input("Press Enter to Show Graph...")

plt.show()

# load the data
#def load_and_preprocess_data(file_path):
#    print("Loading data from {file_path}...")
#    #read csv with custom column names, no header
#    df = pd.read_csv(file_path, names=['date', 'price', 'volume'], header=None)

#    print("Creating datetime index...")
#    df.index = pd.date_range(start='2018-01-01', periods=len(df), freq='H')

#    print("Calculating returns and volatility...")
