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
n_estimators = 50 # Number of boosting iterations
max_depth=10
# Create an AdaBoostRegressor with a DecisionTreeRegressor as the base estimator
ada_regressor = AdaBoostRegressor(
    estimator=DecisionTreeRegressor(max_depth = max_depth),
    n_estimators=n_estimators,
    random_state=42
)

# Fit the AdaBoostRegressor to the training data
ada_regressor.fit(X_train, y_train)

# Make predictions on Train set
ada_y_pred_train = ada_regressor.predict(X_train)
ada_y_mean_squared_error_train = mean_squared_error(y_train, ada_y_pred_train)
ada_y_r2score_train = r2_score(y_train, ada_y_pred_train)
print(f"Number of estimators is: {n_estimators}")
print(f"AdaBoost Regressor Mean Squared Error on Train set: {ada_y_mean_squared_error_train}")
print(f"AdaBoost Regressor R2 Score on Traint set: {ada_y_r2score_train}")


# Make predictions on Test set
ada_y_pred_test = ada_regressor.predict(X_test)
ada_y_mean_squared_error_test = mean_squared_error(y_test, ada_y_pred_test)
ada_y_r2score_test = r2_score(y_test, ada_y_pred_test)
print(f"Number of estimators is: {n_estimators}")
print(f"AdaBoost Regressor Mean Squared Error on Test set: {ada_y_mean_squared_error_test}")
print(f"AdaBoost Regressor R2 Score on Test set: {ada_y_r2score_test}")

# Plotting the Actual vs Predicted values for the AdaBoost Regressor
plt.figure(figsize=(10, 6))

# Scatter plot of actual vs predicted values
plt.scatter(y_test, ada_y_pred_test, alpha=0.5, color='blue', label='AdaBoost Regressor Predictions')

# Ideal fit line (y = x)
line_min = min(y_test.min(), ada_y_pred_test.min())
line_max = max(y_test.max(), ada_y_pred_test.max())
plt.plot([line_min, line_max], [line_min, line_max], 'k--', lw=2, label='Ideal Fit Line')

# Setting labels and title with increased font size
plt.xlabel('Y_test (True)', fontsize=20)  # Increase x-axis label size
plt.ylabel('Y_test (Predicted)', fontsize=20)  # Increase y-axis label size
plt.title('Y_test (True) vs Y_test (Predicted): AdaBoost Regressor', fontsize=19)  # Increase title size

# Enlarging tick labels
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

# Adding legend with enlarged label font size
plt.legend(fontsize=18)  # Adjust legend font size as needed

# Display the plot
plt.grid(True)
plt.show()
