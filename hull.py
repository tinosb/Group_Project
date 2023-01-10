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
hma1  = 30
hma2 = 15
ticks = 750
start_time = time.time()
now_utc = datetime.now(timezone.utc)
date = date.today()
utc_from = datetime.now(timezone.utc).timestamp()
tf = mt5.TIMEFRAME_M1
assets = ['EURUSD']
data_dict=dict((el,[])for el in assets)
#now_utc = datetime.now(pytz.timezone("US/Mountain"))
#utc_from = datetime.now(pytz.timezone("US/Mountain")).timestamp()

mt5.initialize()

for symbol in tqdm(assets):
    try:
        rates_frame = get_data(symbol, ticks, timeframe=tf)
        rates_frame = rates_frame.dropna()
        data_dict[symbol].append(rates_frame)
    except:
        pass

df1 = rates_frame

df1.head()

print(df1['close'])

fast = 9

slow = 18

df_candle['ma_fast'] = df_candle['close'].rolling(window=fast).mean()
df_candle['ma_slow'] = df_candle['close'].rolling(window=slow).mean()