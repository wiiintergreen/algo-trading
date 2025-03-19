#exploring the HMM model for algorithmic trading

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import requests
from datetime import datetime, timedelta
import pytz
import time

print("Starting HMM SP500 analysis...")
print(f"Fetching data from 1 year ago to today")

def fetch_data():
    import requests
    import pandas as pd
    from datetime import datetime, timedelta
    
    # Use Yahoo Finance API directly with a longer time period for better regime detection
    ticker_symbol = "SPY"
    end = int(datetime.now().timestamp())
    start = int((datetime.now() - timedelta(days=252)).timestamp())  # Use 1 year of data
    
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker_symbol}?period1={start}&period2={end}&interval=1d"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        print(f"Downloading historical data for {ticker_symbol}...")
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        data = response.json()
        
        if 'chart' not in data or 'result' not in data['chart'] or not data['chart']['result']:
            print("No data found in response")
            return None, None
            
        result = data['chart']['result'][0]
        timestamps = result['timestamp']
        quotes = result['indicators']['quote'][0]
        
        df = pd.DataFrame({
            'Open': quotes['open'],
            'High': quotes['high'],
            'Low': quotes['low'],
            'Close': quotes['close'],
            'Volume': quotes['volume']
        }, index=pd.to_datetime(timestamps, unit='s'))
        
        if not df.empty:
            print(f"Successfully downloaded {len(df)} days of data")
            return df, ticker_symbol
        else:
            print("Downloaded dataframe is empty")
            return None, None
            
    except Exception as e:
        print(f"Error downloading data: {str(e)}")
        return None, None

if __name__ == "__main__":
    try:
        sp500, successful_ticker = fetch_data()
        
        if sp500 is None or sp500.empty:
            raise ValueError("Could not fetch S&P 500 data")
        
        # Basic data validation
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in sp500.columns for col in required_columns):
            raise ValueError("Downloaded data is missing required columns")
            
        # Convert to Central Time (CST/CDT)
        try:
            sp500.index = sp500.index.tz_localize('UTC').tz_convert('US/Central')
        except Exception as e:
            print(f"Warning: Timezone conversion failed: {e}")
        
        print(f"\nData shape: {sp500.shape}")
        print("\nHere Are The Latest S&P 500 Close Prices:")
        print(sp500['Close'].tail(10))
        
        if len(sp500) > 0:
            print("\nCalculating market features...")
            
            # 1. Returns and volatility features
            sp500['Log_Returns_1d'] = np.log(sp500['Close'] / sp500['Close'].shift(1))
            sp500['Log_Returns_5d'] = sp500['Close'].pct_change(5).apply(np.log1p)
            sp500['Log_Returns_20d'] = sp500['Close'].pct_change(20).apply(np.log1p)
            
            # Simplified volatility calculation for robustness
            sp500['Volatility_5d'] = sp500['Log_Returns_1d'].rolling(window=5, min_periods=3).std()
            sp500['Volatility_20d'] = sp500['Log_Returns_1d'].rolling(window=20, min_periods=10).std()
            
            # 2. Market regime indicators with minimum periods
            sp500['TR'] = np.maximum.reduce([
                sp500['High'] - sp500['Low'],
                abs(sp500['High'] - sp500['Close'].shift(1)),
                abs(sp500['Low'] - sp500['Close'].shift(1))
            ])
            
            # Volume analysis with minimum periods
            sp500['Volume_MA'] = sp500['Volume'].rolling(window=20, min_periods=5).mean()
            sp500['Volume_Ratio'] = sp500['Volume'] / sp500['Volume_MA']
            
            # Market efficiency with shorter window
            sp500['Direction'] = abs(sp500['Close'] - sp500['Close'].shift(10))
            sp500['Path'] = sp500['TR'].rolling(window=10, min_periods=5).sum()
            sp500['MER'] = sp500['Direction'] / sp500['Path']
            
            # Drop NaN values but ensure we have enough data
            sp500 = sp500.dropna()
            
            if len(sp500) < 30:  # Minimum required for meaningful regime detection
                raise ValueError("Insufficient data points after feature calculation")
            
            # Select features for HMM
            feature_columns = [
                'Log_Returns_1d', 'Log_Returns_5d',
                'Volatility_5d', 'Volume_Ratio', 'MER'
            ]
            
            print("\nPreparing HMM model...")
            features = sp500[feature_columns].values
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features)
            
            # Use fixed number of regimes (3) for stability with smaller datasets
            n_components = 3
            print(f"\nUsing {n_components} regimes for stability")
            
            # Train HMM model with more iterations for convergence
            hmm_model = GaussianHMM(
                n_components=n_components,
                covariance_type="diag",  # More stable than full covariance
                n_iter=2000,
                random_state=42,
                tol=1e-5
            )
            
            hmm_model.fit(scaled_features)
            hidden_states = hmm_model.predict(scaled_features)
            sp500['Regime'] = hidden_states
            
            # Analyze regime characteristics
            print("\nRegime Analysis:")
            regime_stats = sp500.groupby('Regime').agg({
                'Log_Returns_1d': ['mean', 'std', lambda x: (x > 0).mean()],
                'Volatility_5d': 'mean',
                'Volume_Ratio': 'mean',
                'MER': 'mean'
            }).round(4)
            
            regime_stats.columns = [
                'Return_Mean', 'Return_Std', 'Win_Rate',
                'Volatility', 'Volume_Intensity',
                'Market_Efficiency'
            ]
            
            print("\nRegime Characteristics:")
            print(regime_stats)
            
            # Calculate regime durations
            durations = []
            current_regime = hidden_states[0]
            current_length = 1
            
            for state in hidden_states[1:]:
                if state == current_regime:
                    current_length += 1
                else:
                    durations.append((current_regime, current_length))
                    current_regime = state
                    current_length = 1
            
            durations.append((current_regime, current_length))
            
            print("\nRegime Analysis:")
            for regime in range(n_components):
                regime_durations = [d for r, d in durations if r == regime]
                if regime_durations:
                    avg_duration = np.mean(regime_durations)
                    print(f"\nRegime {regime}:")
                    print(f"- Average duration: {avg_duration:.1f} days")
                    print(f"- Win rate: {regime_stats.loc[regime, 'Win_Rate']:.1%}")
                    print(f"- Mean daily return: {regime_stats.loc[regime, 'Return_Mean']:.2%}")
                    print(f"- Volatility: {regime_stats.loc[regime, 'Volatility']:.4f}")
            
            # Plot regimes
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), height_ratios=[2, 1])
            
            # Price plot with regimes
            for regime in range(n_components):
                mask = sp500['Regime'] == regime
                ax1.fill_between(sp500.index[mask], sp500['Close'].min(), sp500['Close'].max(),
                               alpha=0.2, label=f'Regime {regime}')
            
            ax1.plot(sp500.index, sp500['Close'], 'k-', linewidth=1)
            ax1.set_ylabel('SPY Price')
            ax1.legend()
            ax1.grid(True)
            
            # Regime probability plot
            regime_probs = hmm_model.predict_proba(scaled_features)
            for i in range(n_components):
                ax2.plot(sp500.index, regime_probs[:, i], 
                        label=f'Regime {i} Prob', alpha=0.7)
            
            ax2.set_ylabel('Regime Probability')
            ax2.legend()
            ax2.grid(True)
            
            plt.tight_layout()
            plt.savefig('data/regime_analysis.png')
            print("\nRegime analysis plot saved as 'regime_analysis.png'")
            
    except Exception as e:
        print(f"Analysis failed: {str(e)}")