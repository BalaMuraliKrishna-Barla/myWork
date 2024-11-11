import pandas as pd
from sklearn.preprocessing import LabelEncoder

data = pd.DataFrame({
    'Color': ['Red', 'Blue', 'Green', 'Blue', 'Red']
})

print(data)

nominalData = pd.DataFrame({'Color': ['Red', 'Blue', 'Green', 'Blue', 'Red']})
ordinalData = pd.DataFrame({'Size': ['Small', 'Medium', 'Large', 'Medium', 'Small']})

encoded_nominalData = pd.get_dummies(nominalData, columns=['Color'])
encoded_ordinalData = pd.get_dummies(ordinalData, columns=['Size'])
print("One Hot Encoding...")
print('Encoded Nominal Data \n', encoded_nominalData)
print('\n\nEncoded Ordinal Data\n', encoded_ordinalData)

label_encoder = LabelEncoder()

nominalData['encodedColor'] = label_encoder.fit_transform(nominalData['Color'])
ordinalData['encodedSize'] = label_encoder.fit_transform(ordinalData['Size'])
print("Label Encoding....")
print('On nominal data\n', nominalData)
print('\n\nOn ordinal data\n', ordinalData)

from sklearn.preprocessing import MinMaxScaler

data1 = pd.DataFrame({
    'feature1': [1, 2, 3, 4, 5],
    'feature2': [256, 354, 300, 490, 650]
})
print(data1)
scaler = MinMaxScaler()
scaled_data = pd.DataFrame(scaler.fit_transform(data1), columns=data1.columns)
print(scaled_data)

from sklearn.preprocessing import StandardScaler
standard_scaler = StandardScaler()
scaled_data = standard_scaler.fit_transform(data1)
res = pd.DataFrame(scaled_data, columns=data1.columns)
print(res)

from sklearn.model_selection import train_test_split

data = pd.DataFrame({
    'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'feature2': [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    'label': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
})
df = pd.DataFrame(data)
x = df[['feature1', 'feature2']]
y = df['label']
x_train, x_test, y_train, y_test = train_test_split(x, y)
print('X training set : \n', x_train)
print('X testing set : \n', x_test)
print('Y training set : \n', y_train)
print('Y testing set : \n', y_test)
