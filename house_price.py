from sklearn.linear_model import LinearRegression
import numpy as np

# Area data
X = np.array([[1000], [1500], [2000], [2500], [3000]])

# Price data
y = np.array([20, 30, 40, 50, 60])

# Create model
model = LinearRegression()

# Train model
model.fit(X, y)

# Predict house price
prediction = model.predict([[2200]])

print("Predicted House Price:", prediction[0], "Lakhs")