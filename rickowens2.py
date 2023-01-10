import pandas as pd; import csv as cs; from datetime import date
from pandas import DataFrame as dat1
import MetaTrader5 as mt5; from tqdm import tqdm; import pytz; from datetime import datetime,timezone
import plotly.graph_objects as go; from plotly.subplots import make_subplots; import plotly
from gucci import hull_moving_average, get_data, moving_average_crossover, relative_strength_index, moneyflowindex; import numpy as np
import time
from ta.volatility import AverageTrueRange as ATR1

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

#Variables
n1 = 14
rsiob = 70
rsios = 30
short_per = 10
long_per = 25
rsiper = 7
hma1  = 7
hma2 = 15
mfiob = 50
mfios = 10
fastema = 9
slowema = 18
hkopen = 16
hkhigh = 16
hklow = 16
hkclose = 16
ticks = 2000
start_time = time.time()
#now_utc = datetime.now(timezone.utc)
date = date.today()
#utc_from = datetime.now(timezone.utc).timestamp()
tf = mt5.TIMEFRAME_M5
assets = ['EURUSD']
data_dict=dict((el,[])for el in assets)
now_utc = datetime.now(pytz.timezone("US/Mountain"))
utc_from = datetime.now(pytz.timezone("US/Mountain")).timestamp()

mt5.initialize()

for symbol in tqdm(assets):
    try:
        rates_frame = get_data(symbol, ticks, timeframe=tf)
        rates_frame = rates_frame.dropna()
        data_dict[symbol].append(rates_frame)
    except:
        pass

df1 = rates_frame







def df2(df=df1):
    df['1HMA'] = hull_moving_average(df['close'], n=hma1)
    df['2HMA'] = hull_moving_average(df['close'], n=hma2)
    df['HMA1'] = hull_moving_average(df['close'], n=hma1)
    df['HMA2'] = hull_moving_average(df['close'], n=hma2)
    df['HO'] = hull_moving_average(df['open'], n=hkopen)
    df['HH'] = hull_moving_average(df['high'], n=hkhigh)
    df['HL'] = hull_moving_average(df['low'], n=hklow)
    df['HC'] = hull_moving_average(df['close'], n=hkclose)
    df['CRSIG'] = moving_average_crossover(df, short_per, long_per)
    df['RSIVAL'] = relative_strength_index(df, rsiper)
    df['volume'] = df['tick_volume']
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['prev_typical_price'] = df['typical_price'].shift(1)
    df['prev_close'] = df['close'].shift(1)
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = abs(df['high'] - df['close'].shift(1))
    df['tr3'] = abs(df['low'] - df['close'].shift(1))
    df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    df["TR"] = df[["high", "low", "close"]].apply(lambda x: max(x) - min(x), axis=1)
    atr = df["TR"].rolling(14).mean()
    df["ATR"] = atr
    df["mfi"] = moneyflowindex(df)
    df['ma_fast'] = df['close'].rolling(window=fastema).mean()
    df['ma_slow'] = df['close'].rolling(window=slowema).mean()
    df['signal'] = np.where(df['1HMA'] > df['2HMA'], 1, 0)
    df['signal1'] = np.where(df['RSIVAL'] > rsiob, 1, 0)
    df['signal2'] = np.where(df['RSIVAL'] < rsios, 1, 0)
    df['signal3'] = np.where(df['mfi'] > mfiob, 1, 0)
    df['signal4'] = np.where(df['mfi'] < mfios, 1, 0)
    df['position'] = df['signal'].diff()
    df['position1'] = df['signal1'].diff()
    df['position2'] = df['signal2'].diff()
    df['position3'] = df['signal3'].diff()
    df['position4'] = df['signal4'].diff()
    
    return df
    
df2(df1)

print(df1['signal'],df1['signal1'],df1['signal2'],df1['signal3'],df1['signal4'])

position = {1:["BUY", "#228B22", -10], -1:["SELL", "#FF0000", 10]}



with open('coolswag1.csv', 'w', newline='') as csv_file:
    df1.to_csv(path_or_buf=csv_file)
    

df = pd.read_csv('coolswag1.csv')

fig = make_subplots(rows=7, cols=3)  



fig.append_trace(
    go.Candlestick(
        x=df['time'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        increasing_line_color='#ff9900',
        decreasing_line_color='black',
        showlegend=False
    ), row=1, col=1  # <------------ upper chart
)

fig.append_trace(
    go.Candlestick(
        x=df['time'],
        open=df['HO'],
        high=df['HH'],
        low=df['HL'],
        close=df['HC'],
        increasing_line_color='#ff9900',
        decreasing_line_color='black',
        showlegend=False,
        name='Hull Smoothed candles'
    ), row=1, col=2  # <------------ upper chart
)
# price Line




fig.append_trace(
    go.Scatter(
        x=df['time'],
        y=df['1HMA'],
        line=dict(color='#ff9900', width=3),
        name='Hull',
        mode='lines'
    ), row=1, col=1  # <------------ upper chart
)

fig.append_trace(
    go.Scatter(
        x=df['time'],
        y=df['2HMA'],
        line=dict(color='#ff9900', width=3),
        name='Hull2',
        mode='lines'
    ), row=1, col=1  # <------------ upper chart
)

candles_data = go.Candlestick(
        x=df['time'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],)


ma11=go.Scatter(
        x=df['time'],
        y=df['1HMA'],
        line=dict(color='#ff9900', width=3),
        name='Hull',
        mode='lines'
    )

ma22=go.Scatter(
        x=df['time'],
        y=df['2HMA'],
        line=dict(color='#ff9900', width=3),
        name='Hull2',
        mode='lines'
    )

candlestick_chart = go.Figure(data=[candles_data, ma11, ma22])

increment = 0

for trade in df1.position:
    if trade == 1 or trade == -1:
        candlestick_chart.add_annotation(
            x=df1.index[increment],
            y=df1.ma_slow[increment],
            xref="x",
            yref="y",
            text=position[trade][0],
            showarrow=True,
            font=dict(
                family="Courier New, monospace",
                size=10,
                color="#ffffff"
            ),
        align="center",
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="#636363",
        ax=position[trade][2],
        ay=0,
        bordercolor="#c7c7c7",
        borderwidth=2,
        borderpad=4,
        bgcolor=position[trade][1],
        opacity=0.8)
    increment +=1

candlestick_chart.update_layout(xaxis_rangeslider_visible=False)

candlestick_chart.show()

fig.append_trace(
    go.Scatter(
        x=df.index,
        y=df['RSIVAL'],
        line=dict(color='#ff9900', width=2),
        name='RSI',
    ), row=4, col=1  #  <------------ lower chart
)

fig.append_trace(
   go.Scatter(
        x=df.index,
        y=df['mfi'],
        line=dict(color='#ff9900', width=2),
        name='mfi',
    ), row=5, col=1  #  <------------ lower chart
)

fig.append_trace(
    go.Scatter(
        x=df['time'],
        y=df['CRSIG'],
        #line=dict(color='#ff9900', width=100),
        name='CRSIG',
    ), row=6, col=1  #  <------------ lower chart
)

fig.append_trace(
    go.Scatter(
        x=df.index,
        y=df['ATR'],
        #line=dict(color='#ff9900', width=100),
        name='atr',
    ), row=7, col=1  #  <------------ lower chart
)

fig.append_trace(
    go.Scatter(
        x=df['time'],
        y=df['ma_fast'],
        #line=dict(color='#ff9900', width=100),
        name='fastma',
    ), row=1, col=2  #  <------------ lower chart
)

fig.append_trace(
    go.Scatter(
        x=df['time'],
        y=df['ma_slow'],
        #line=dict(color='#ff9900', width=100),
        name='maslow',
    ), row=1, col=2  #  <------------ lower chart
)
# Slow signal (%d)
#fig.append_trace(
#    go.Scatter(
#        x=df.index,
#        y=df['stochd_14_3_3'],
#        line=dict(color='#000000', width=2),
#        name='slow'
#    ), row=2, col=1  #<------------ lower chart
#)
# Extend our y-axis a bit
fig.update_yaxes(range=[-2, 2], row=6, col=1)
fig.update_yaxes(range=[-20, 120], row=4, col=1)
fig.update_yaxes(range=[0, 60], row=5, col=1)
# Add upper/lower bounds
#fig.add_hline(y=0, col=1, row=2, line_color="#666", line_width=2)
#fig.add_hline(y=100, col=1, row=2, line_color="#666", line_width=2)
# Add overbought/oversold
#fig.add_hline(y=20, col=1, row=2, line_color='#336699', line_width=2, line_dash='dash')
#fig.add_hline(y=80, col=1, row=2, line_color='#336699', line_width=2, line_dash='dash')
# Make it pretty
layout = go.Layout(
    plot_bgcolor='#efefef',
    # Font Families
    font_family='Monospace',
    font_color='#000000',
    font_size=10,
    xaxis=dict(
        rangeslider=dict(
            visible=True
        )
    )
)
fig.update_layout(layout)
# View our chart in the system default HTML viewer (Chrome, Firefox, etc.)
fig.show()

print("--- %s seconds ---" % (time.time() - start_time))


pip_cost = .08
lot_size = 10

begin_prices = []
end_prices = []
profits = 0
profit_collect = [0]

for i, row in df1.iterrows():
    if row['position'] == 1:
        begin_prices.append(float(row['open']))
        if row['position'] == 1:
            end_prices.append(float(row['close']))

for i in np.arange(len(begin_prices)):
    profit = (end_prices[i] - begin_prices[i]) * 100000 * pip_cost * lot_size
    profits += profit
    profit_collect.append(profits)

print("return: " + str(int(profits)))

profitline=go.Scatter(
        x=df['time'],
        y=profit_collect,
        line=dict(color='#ff9900', width=3),
        name='Hull',
        mode='lines'
    )

candlechart = go.Figure(data=[profitline])
candlechart.show()

