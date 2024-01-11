import pandas as pd

data = pd.read_csv('framingham.csv')

data.drop(data.columns[data.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)

data.dropna().to_csv('cleaned.csv')