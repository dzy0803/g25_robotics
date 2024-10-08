import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

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

# Decision Tree Regressor
tree = DecisionTreeRegressor(max_depth=20, splitter='best')
tree.fit(X_train, y_train)
decision_tree_y_pred = tree.predict(X_test)
decision_tree_y_mean_squared_error = mean_squared_error(y_test, decision_tree_y_pred)
decision_tree_y_r2score = r2_score(y_test, decision_tree_y_pred)

print(f"Decision Tree Regressor Mean Squared Error: {decision_tree_y_mean_squared_error}")
print(f"Decision Tree Regressor R2 Score: {decision_tree_y_r2score}")

# Polynomial Regression
poly = PolynomialFeatures(degree=10)
X_poly_train = poly.fit_transform(X_train)
X_poly_test = poly.transform(X_test)
poly_regressor = LinearRegression()
poly_regressor.fit(X_poly_train, y_train)
poly_y_pred = poly_regressor.predict(X_poly_test)
poly_y_mean_squared_error = mean_squared_error(y_test, poly_y_pred)
poly_y_r2score = r2_score(y_test, poly_y_pred)

print(f"Polynomial Regression Mean Squared Error: {poly_y_mean_squared_error}")
print(f"Polynomial Regression R2 Score: {poly_y_r2score}")

# Plotting the Actual vs Predicted values for both models in a single plot
plt.figure(figsize=(10, 5))

# Scatter plots for Decision Tree Regressor and Polynomial Regression
plt.scatter(y_test, decision_tree_y_pred, alpha=0.5, color='blue', label='Decision Tree Regressor')
plt.scatter(y_test, poly_y_pred, alpha=0.5, color='red', label='Polynomial Regression')

# Ideal fit line (y = x)
line_min = min(y_test.min(), decision_tree_y_pred.min(), poly_y_pred.min())
line_max = max(y_test.max(), decision_tree_y_pred.max(), poly_y_pred.max())
plt.plot([line_min, line_max], [line_min, line_max], 'k--', lw=2, label='Ideal Fit Line')

# Polynomial Regression best-fit line
poly_coeff = np.polyfit(y_test, poly_y_pred, deg=1)  # Fit a linear best-fit line
poly_best_fit_line = np.poly1d(poly_coeff)
plt.plot(np.sort(y_test), poly_best_fit_line(np.sort(y_test)), color='grey', linestyle='--', linewidth=2, label='Polynomial Best Fit Line')

plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted: Decision Tree Regressor vs Polynomial Regression')
plt.legend()
plt.show()
