import pandas as pd
import quandl
import math

df = quandl.get('WIKI/GOOGL')

print('Before REGRESSION applied: ')
print(df.head())

df = df[['Adj. Open','Adj. Low','Adj. High','Adj. Close','Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close']*100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open']*100.0

df = df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]

print('After REGRESSION applied: ')
print(df.head())

forecast_col = 'Adj.Close'
df.fillna(-99999, inplace = True)
forecast_out = int(math.ceil(0.1*len(df)))

df['label'] = df[forecast_col].shift(-forecast_out)

print('After forecast: ')
df.dropna(inplace = True)
print(df.tail())
