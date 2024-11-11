import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

X = np.random.rand(100, 1) * 10 
y = 5 * X + np.random.randn(100, 1) * 2 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

plt.scatter(X_test, y_test, color='blue', label='True values')
plt.plot(X_test, y_pred, color='red', label='Predicted values')
plt.xlabel('Number of Rooms')
plt.ylabel('House Price')
plt.title('Linear Regression: House Prices')
plt.legend()
plt.show()
