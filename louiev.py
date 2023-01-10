import matplotlib.pyplot as plt
import pandas as pd
import pandas_ta as ta
from mpl_finance import candlestick_ohlc

# Initialize the matplotlib figure and subplots
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

while True:
    # Load the latest data and create a Pandas DataFrame
    df = pd.read_csv('coolswag1.csv')

    # Calculate the RSI using pandas_ta
    df['RSI'] = ta.momentum.rsi(df['close'])

    # Update the candlestick chart
    candlestick_ohlc(ax1, df[['time', 'open', 'high', 'low', 'close']].values, width=1, colorup='g', colordown='r')

    # Update the RSI chart
    ax2.plot(df['Date'], df['RSI'])

    # Redraw the figure
    plt.draw()

    # Clear the figure
    ax1.clear()
    ax2.clear()

    # Wait for a while before updating the chart again
    time.sleep(60)
