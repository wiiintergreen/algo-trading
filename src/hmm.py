#exploring the HMM model for algorithmic trading

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import yfinance as yf
from datetime import datetime, timedelta
import pytz
import time

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

# Calculate daily returns & volatility
sp500['Log_Returns'] = (sp500['Close']
    .pct_change()
    .apply(np.log))
sp500.dropna(subset=[('Log_Returns', '')], inplace=True) # Drops NaN values in Log_Returns to avoid issues with calculating volatility

sp500['Volatility'] = (sp500['Log_Returns']
    .rolling(window=15)
    .std()) 
sp500.dropna(subset=[('Volatility', '')], inplace=True) # Drops NaN values in Volatility to avoid issues scaling features

print(sp500.head())

# Count the number of NaN values in 'Volatility' column
nan_count = sp500['Log_Returns'].isna().sum()

# Count the number of non-NaN values in 'Volatility' column
non_nan_count = sp500['Log_Returns'].notna().sum()

print(f"The 'Log Retruns column contains {nan_count} NaN values and {non_nan_count} non-NaN values.")

# Prepare features for HMM
features = sp500[[('Log_Returns', ''), ('Volatility', '')]].values

# Standardize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Train the HMM model
n_components = 3  # Number of regimes
hmm_model = GaussianHMM(n_components=n_components, covariance_type="full", n_iter=1000)
hmm_model.fit(scaled_features)

# Predict hidden states (regimes)
hidden_states = hmm_model.predict(scaled_features)
sp500['Regime'] = hidden_states

# Plot both volatility and price on the same figure
fig, ax1 = plt.subplots(figsize=(24, 12))

# Plot price on the first y-axis
ax1.set_xlabel("Date", fontsize=20)
ax1.set_ylabel("Price (Close)", color='tab:blue', fontsize=20)
ax1.plot(sp500.index, sp500['Close'], color='tab:blue', label="Price")
ax1.tick_params(axis='y', labelcolor='tab:blue', labelsize=16)

# Create a second y-axis to plot volatility
ax2 = ax1.twinx()
ax2.set_ylabel("Volatility", color='tab:orange', fontsize=20)
ax2.plot(sp500.index, sp500['Volatility'], color='tab:orange', label="Volatility")
ax2.tick_params(axis='y', labelcolor='tab:orange', labelsize=16)

# Add hidden state regions to the plot
for state in range(n_components):
    state_mask = sp500['Regime'] == state
    ax1.fill_between(sp500.index, sp500['Close'].min(), sp500['Close'].max(),
                     where=state_mask, alpha=0.1, label=f"Regime {state}")

# Format the x-axis
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax1.xaxis.set_major_locator(mdates.MinuteLocator(interval=30))
ax1.tick_params(axis='x', labelsize=16)

# Add title and legend
plt.title("S&P 500 1m Price, Volatility, and Regimes")
fig.tight_layout()

input("Press Enter to Show Graph...")
plt.legend(loc="upper left")
plt.show()

# Save the figure to data folder as PNG
fig.savefig('data/plot_with_hmm.png')

# Print regime-based summary statistics
print("\nRegime-Based Summary Statistics:")
print(sp500.groupby('Regime').mean())