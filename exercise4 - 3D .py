import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

# Generate synthetic data
x1 = np.arange(0, 10, 0.1)
x2 = np.arange(0, 10, 0.1)
x1, x2 = np.meshgrid(x1, x2)
y = np.sin(x1) * np.cos(x2) + np.random.normal(scale=0.1, size=x1.shape)

# Flatten the arrays
x1_flat = x1.flatten()
x2_flat = x2.flatten()
y_flat = y.flatten()
X = np.vstack((x1_flat, x2_flat)).T

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_flat, test_size=0.3, random_state=42)

# Create models
ada_tree = AdaBoostRegressor(estimator=DecisionTreeRegressor(max_depth=5),n_estimators=100, learning_rate=1,loss='square')
decision_tree = DecisionTreeRegressor()

# Fit the models
ada_tree.fit(X_train, y_train)
decision_tree.fit(X_train, y_train)

# Predict on the test set
y_pred_ada_tree = ada_tree.predict(X_test)
y_pred_decision_tree = decision_tree.predict(X_test)

# Calculate R2 scores
ada_tree_score = r2_score(y_test, y_pred_ada_tree)
decision_tree_score = r2_score(y_test, y_pred_decision_tree)

mse_ada_tree = mean_squared_error(y_test, y_pred_ada_tree)
mse_decision_tree = mean_squared_error(y_test, y_pred_decision_tree)

print(f"AdaBoost Regressor R2 Score: {ada_tree_score}")
print(f"Decision Tree Regressor R2 Score: {decision_tree_score}")
print(f"AdaBoost Regressor Mean Squared Error: {mse_ada_tree}")
print(f"Decision Tree Regressor Mean Squared Error: {mse_decision_tree}")

# Plot the predictions for AdaBoost Regressor
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(121, projection='3d')
ax.scatter(X_test[:, 0], X_test[:, 1], y_test, color='blue', label='True Values', s=10)
ax.scatter(X_test[:, 0], X_test[:, 1], y_pred_ada_tree, color='red', label='Predictions', s=10)
ax.set_title('AdaBoost Regressor')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Y')
ax.legend()

# Plot the predictions for Decision Tree Regressor
ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(X_test[:, 0], X_test[:, 1], y_test, color='blue', label='True Values', s=10)
ax2.scatter(X_test[:, 0], X_test[:, 1], y_pred_decision_tree, color='red', label='Predictions', s=10)
ax2.set_title('Decision Tree Regressor')
ax2.set_xlabel('X1')
ax2.set_ylabel('X2')
ax2.set_zlabel('Y')
ax2.legend()

plt.tight_layout()
plt.show()
