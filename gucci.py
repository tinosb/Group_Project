import numpy as np
import MetaTrader5 as mt5
import pandas as pd
from datetime import timezone, datetime

utc_from = datetime.now(timezone.utc).timestamp()

def get_data(symbol, n, timeframe):
    rates = mt5.copy_rates_from(symbol, timeframe, utc_from, n)
    rates_frame = pd.DataFrame(rates)
    rates_frame['time'] = pd.to_datetime(rates_frame['time'], unit='s')
    rates_frame = rates_frame.set_index('time')
    # rates_frame = rates_frame.drop(columns=['tick_volume','real_volume','spread'])
    return rates_frame


#This function takes in two arguments: data, which is a list or array of data points, and period, which is the number of data points to use in the moving average calculation. It returns the Hull moving average of the data.

#To use this function, you would call it with the desired data and period, like this:

#hma_value = hma(data, period)
#This will calculate the Hull moving average of the data using the specified period and return the result.

# Calculate Crossover

def moving_average_crossover(df, short_per, long_per):
    # Calculate the moving averages
    short_ma = hull_moving_average(df['close'], short_per)
    long_ma = hull_moving_average(df['close'], long_per)
    
    # Shift the moving averages so that they can be compared
    short_ma_shifted = short_ma.shift(1)
    long_ma_shifted = long_ma.shift(1)
    
    # Create a DataFrame to store the buy and sell signals
    signals = pd.DataFrame(index=df.index)
    signals['signal'] = 0.0
    
    # Set the signal to 1 when the short moving average crosses above the long moving average
    signals.loc[short_ma > long_ma, 'signal'] = 1.0
    
    # Set the signal to -1 when the short moving average crosses below the long moving average
    signals.loc[short_ma < long_ma, 'signal'] = -1.0
    
    return signals

def relative_strength_index(df, n):
    # Calculate the change in price between each period
    delta = df['close'].diff()
    
    # Create a DataFrame to store the gain and loss
    gain_loss = pd.DataFrame(index=df.index)
    gain_loss['gain'] = 0.0
    gain_loss['loss'] = 0.0
    
    # Identify the periods where the price increased, and calculate the average gain over the specified number of periods
    gain_loss.loc[delta > 0, 'gain'] = delta
    avg_gain = gain_loss['gain'].rolling(n).mean()
    
    # Identify the periods where the price decreased, and calculate the average loss over the specified number of periods
    gain_loss.loc[delta < 0, 'loss'] = -delta
    avg_loss = gain_loss['loss'].rolling(n).mean()
    
    # Calculate the relative strength
    rs = avg_gain / avg_loss
    
    # Calculate the RSI
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calc_mfi(row):
  typical_price = row['typical_price']
  volume = row['volume']
  prev_typical_price = row['prev_typical_price']
  
  if typical_price > prev_typical_price:
    money_flow = typical_price * volume
  else:
    money_flow = 0
  
  mf = money_flow / sum(row['typical_price'] * row['volume'])
  mfi = 100 - (100 / (1 + mf))
  
  return mfi



def hull_moving_average(df, n):
    # Calculate the exponential moving average
    ema = df.ewm(span=n//2).mean()
    
    # Square the EMA
    ema_sq = ema**2
    
    # Calculate the EMA of the squared EMA
    ema_sq_ema = ema_sq.ewm(span=n//2).mean()
    
    # Calculate the square root of the EMA of the squared EMA
    ema_sq_ema_sqrt = ema_sq_ema**0.5
    
    # Calculate the EMA of the square root of the EMA of the squared EMA
    hma = ema_sq_ema_sqrt.ewm(span=n//2).mean()
    
    return hma


def moneyflowindex(df):
    df["TP"] = (df["high"] + df["low"] + df["close"]) / 3
    df["MF"] = df["TP"] * df["volume"]
    df["PMF"] = df["MF"].where(df["close"] > df["close"].shift(1), 0)
    df["NMF"] = df["MF"].where(df["close"] < df["close"].shift(1), 0)
    n = 14  # Number of periods
    pmf_sum = df["PMF"].rolling(n).sum()
    nmf_sum = df["NMF"].rolling(n).sum()
    mfi = 100 - (100 / (1 + (pmf_sum / nmf_sum)))
    df["MFI"] = mfi
    return mfi



#def moving_average_crossover(df, short_per, long_per):
#    # Calculate the moving averages
#    short_ma = hull_moving_average(df['close'], short_per)
#    long_ma = hull_moving_average(df['close'], long_per)
#    
#    # Shift the moving averages so that they can be compared
#    short_ma_shifted = short_ma.shift(1)
#    long_ma_shifted = long_ma.shift(1)
#    
#    # Create a DataFrame to store the buy and sell signals
#    signals = pd.DataFrame(index=df.index)
#    signals['signal'] = 0.0
#    
#   # Set the signal to 1 when the short moving average crosses above the long moving average
#    signals.loc[short_ma > long_ma, 'signal'] = 1.0
#    
#    # Set the signal to -1 when the short moving average crosses below the long moving average
#    signals.loc[short_ma < long_ma, 'signal'] = -1.0
#    
#    return signals