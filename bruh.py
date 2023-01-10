import MetaTrader5 as mt5
import plotly.graph_objects as go
import pandas as pd
import time

# Connect to MT5
mt5.initialize()

# Set the symbol and time frame
symbol = "EURUSD"
time_frame = mt5.TIMEFRAME_M1

# Create a Figure object
fig = go.Figure()

while True:
    # Request the data
    rates = mt5.copy_rates_from_pos(symbol, time_frame, 0, 1000)

    # Convert the data to a pandas DataFrame
    df = pd.DataFrame(rates)

    # Convert the 'time' column to datetime
    df['time'] = pd.to_datetime(df['time'], unit='s')

    # Set the 'time' column as the index
    df.set_index(df['time'], inplace=True)

    # Update the Figure object with the new data
    fig.update_traces(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close']))
    fig.show()
    # Wait a few seconds before updating the chart again
    time.sleep(3)



# Disconnect from MT5
mt5.shutdown()
