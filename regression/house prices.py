import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Load data
data = pd.read_csv("Housing.csv")

# Separate features and target
x = data.iloc[:, 1:].values
y = data.iloc[:, 0].values

# Define ColumnTransformer
transform = ColumnTransformer(
    transformers=[
        ('mainroad', OneHotEncoder(), [4]), 
        ('guestroom', OneHotEncoder(), [5]),
        ('basement', OneHotEncoder(), [6]),
        ('hotwaterheating', OneHotEncoder(), [7]),
        ('airconditioning', OneHotEncoder(), [8]),
        ('parking', OneHotEncoder(), [9]), 
        ('prefarea', OneHotEncoder(), [10]),
        ('furningstatus', OneHotEncoder(), [11])
    ],
    remainder='passthrough'
)

# Apply transformations
x_transformed = transform.fit_transform(x)

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_transformed, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LinearRegression()
model.fit(x_train, y_train)

# Predict on the test set
y_pred = model.predict(x_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"R-squared: {r2}")
