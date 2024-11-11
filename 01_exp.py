import pandas as pd

data = pd.read_csv('../sampleData.csv')
print(data.head())

x = data.iloc[:,:-1]
y = data.iloc[:,-1]
print()
print('independent variables : \n', x)
print()
print('dependent variables : \n', y)
print()

missing_values = data.isnull().sum()
print('missing values : \n', missing_values)
print()

cols = data.columns.tolist()
for col in cols:
    if data[col].dtype == 'int64' or data[col].dtype == 'float64':
        data
data = data.fillna(data.mean())

missing_values = data.isnull().sum()
print('missing values : \n', missing_values)
print(data)
