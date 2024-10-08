import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Generate synthetic data
x1 = np.arange(0, 10, 0.1)
x2 = np.arange(0, 10, 0.1)
x1, x2 = np.meshgrid(x1, x2)
y = np.sin(x1) * np.cos(x2) + np.random.normal(scale=0.1, size=x1.shape)

# Flatten the arrays
x1 = x1.flatten()
x2 = x2.flatten()
y = y.flatten()
X = np.vstack((x1, x2)).T

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Set parameters
n_estimators = 75  # Number of boosting iterations
max_depth=20
# Create an AdaBoostRegressor with a DecisionTreeRegressor as the base estimator
ada_regressor = AdaBoostRegressor(
    estimator=DecisionTreeRegressor(max_depth = max_depth),
    n_estimators=n_estimators,
    random_state=42
)

# Fit the AdaBoostRegressor to the training data
ada_regressor.fit(X_train, y_train)

# Make predictions
ada_y_pred = ada_regressor.predict(X_test)
ada_y_mean_squared_error = mean_squared_error(y_test, ada_y_pred)
ada_y_r2score = r2_score(y_test, ada_y_pred)

print(f"Number of estimators is: {n_estimators}")
print(f"AdaBoost Regressor Mean Squared Error: {ada_y_mean_squared_error}")
print(f"AdaBoost Regressor R2 Score: {ada_y_r2score}")

# Plotting the Actual vs Predicted values for the AdaBoost Regressor
plt.figure(figsize=(10, 6))

# Scatter plot of actual vs predicted values
plt.scatter(y_test, ada_y_pred, alpha=0.5, color='blue', label='AdaBoost Regressor Predictions')

# Ideal fit line (y = x)
line_min = min(y_test.min(), ada_y_pred.min())
line_max = max(y_test.max(), ada_y_pred.max())
plt.plot([line_min, line_max], [line_min, line_max], 'k--', lw=2, label='Ideal Fit Line')

# Setting labels and title
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values: AdaBoost Regressor')

# Adding legend
plt.legend()

# Display the plot
plt.grid(True)
plt.show()
