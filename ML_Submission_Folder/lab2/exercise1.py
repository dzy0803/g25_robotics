import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# 1. Pre-processing
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
# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# 2.Implement decision tree train
tree = DecisionTreeRegressor(max_depth=10, splitter="best")  # max_depth = {5, 10} and splitter = {"best", "random"} used
tree.fit(X_train, y_train)
# Testing
decision_tree_y_pred_test = tree.predict(X_test)     # Testing for test set prediction
decision_tree_y_pred_train = tree.predict(X_train)   # Testing for train set prediction
# test set prediction assess
decision_tree_y_mean_squared_error_Test = mean_squared_error(y_test, decision_tree_y_pred_test)
decision_tree_y_r2score_Test = r2_score(y_test, decision_tree_y_pred_test)
print(f"Decision Tree Regression - Test set MSE: {decision_tree_y_mean_squared_error_Test:.4f}")
print(f"Decision Tree Regression - Test set R^2: {decision_tree_y_r2score_Test:.4f}")
# train set prediction assess
decision_tree_y_mean_squared_error_Train = mean_squared_error(y_train, decision_tree_y_pred_train)
decision_tree_y_r2score_Train = r2_score(y_train, decision_tree_y_pred_train)
print(f"Decision Tree Regression - Trian set MSE: {decision_tree_y_r2score_Train :.4f}")
print(f"Decision Tree Regression - Train set R^2: {decision_tree_y_r2score_Train:.4f}")
# 3D Decision Tree Visualization
# Plot configuration
fig = plt.figure(figsize=(15, 10))
# 3D Plot for train set
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(X_train[:, 0], X_train[:, 1], y_train, color='r', label='Y_train (True)', s=20)
ax1.scatter(X_train[:, 0], X_train[:, 1], decision_tree_y_pred_train, color='b', label='Y_train (Predicted)', s=20, alpha=0.5)
ax1.set_xlabel('X1 (Train)')
ax1.set_ylabel('X2 (Train)')
ax1.set_zlabel('Y')
ax1.set_title('Decision Tree: 3D Visualization of X1_train, X2_train, Y_train, Y_train_pred')
ax1.legend()
# 3D Plot for test set
ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(X_test[:, 0], X_test[:, 1], y_test, color='r', label='Y_test (True)', s=20)
ax2.scatter(X_test[:, 0], X_test[:, 1], decision_tree_y_pred_test, color='b', label='Y_test (Predicted)', s=20, alpha=0.5)
ax2.set_xlabel('X1 (Test)')
ax2.set_ylabel('X2 (Test)')
ax2.set_zlabel('Y')
ax2.set_title('Decision Tree: 3D Visualization of X1_test, X2_test, Y_test, Y_test_pred')
ax2.legend()
plt.show()

# Scatter plots of the Actual vs Predicted values for both Train and Test set
plt.figure(figsize=(20, 10))

# Train set
plt.subplot(1, 2, 1)  # First plot
plt.scatter(y_train, decision_tree_y_pred_train, alpha=0.5, color='blue', label='Decision Tree Regressor')
# Ideal fit line (y = x)
line_min = min(y_train.min(), decision_tree_y_pred_train.min())
line_max = max(y_train.max(), decision_tree_y_pred_train.max())
plt.plot([line_min, line_max], [line_min, line_max], 'k--', lw=2, label='Ideal Fit Line')
plt.xlabel('Y_train (True)', fontsize=20)  # Increase x-axis label size
plt.ylabel('Y_train (Predicted)', fontsize=20)  # Increase y-axis label size
plt.title('Y_train (True) vs. Y_train (Predicted) -- Decision Tree Regressor', fontsize=18)  # Increase title size
plt.xticks(fontsize=20)  # Increase x-tick label size
plt.yticks(fontsize=20)  # Increase y-tick label size
plt.legend(fontsize=20)  # Increase legend font size

# Test set
plt.subplot(1, 2, 2)  # Second plot
plt.scatter(y_test, decision_tree_y_pred_test, alpha=0.5, color='grey', label='Decision Tree Regressor')
# Ideal fit line (y = x)
line_min = min(y_test.min(), decision_tree_y_pred_test.min())
line_max = max(y_test.max(), decision_tree_y_pred_test.max())
plt.plot([line_min, line_max], [line_min, line_max], 'k--', lw=2, label='Ideal Fit Line')
plt.xlabel('Y_test (True)', fontsize=20)  # Increase x-axis label size
plt.ylabel('Y_test (Predicted)', fontsize=20)  # Increase y-axis label size
plt.title('Y_test (True) vs. Y_test (Predicted) -- Decision Tree Regressor', fontsize=18)  # Increase title size
plt.xticks(fontsize=20)  # Increase x-tick label size
plt.yticks(fontsize=20)  # Increase y-tick label size
plt.legend(fontsize=20)  # Increase legend font size

plt.tight_layout()
plt.show()


# 3.Implement polynomial regression train
poly = PolynomialFeatures(degree=10)
X_poly_train = poly.fit_transform(X_train)
X_poly_test = poly.transform(X_test)
poly_regressor = LinearRegression()
poly_regressor.fit(X_poly_train, y_train)
# Testing 
poly_y_pred_test = poly_regressor.predict(X_poly_test)# Testing for test set prediction
poly_y_pred_train = poly_regressor.predict(X_poly_train)# Testing for train set prediction
# test set prediction assess
poly_y_mean_squared_error_test = mean_squared_error(y_test, poly_y_pred_test)
poly_y_r2score_test = r2_score(y_test, poly_y_pred_test)
print(f"Polynomial Regression Mean Squared Error for test set: {poly_y_mean_squared_error_test}")
print(f"Polynomial Regression R2 Score for test set: {poly_y_r2score_test}")
# train set prediction assess
poly_y_mean_squared_error_train = mean_squared_error(y_train, poly_y_pred_train)
poly_y_r2score_train= r2_score(y_train, poly_y_pred_train)
print(f"Polynomial Regression Mean Squared Error for train set: {poly_y_mean_squared_error_train}")
print(f"Polynomial Regression R2 Score for train set: {poly_y_r2score_train}")



#4. Comparision Plot Configuration
# Plotting the Actual vs Predicted values for both models in a single plot
plt.figure(figsize=(10, 5))
# pScatter plots for Decision Tree Regressor and Polynomial Regression
plt.scatter(y_test, decision_tree_y_pred_test, alpha=0.5, color='blue', label='Decision Tree Regressor')
plt.scatter(y_test, poly_y_pred_test, alpha=0.5, color='red', label='Polynomial Regression')
# Ideal fit line (y = x)
line_min = min(y_test.min(), decision_tree_y_pred_test.min(), poly_y_pred_test.min())
line_max = max(y_test.max(), decision_tree_y_pred_test.max(), poly_y_pred_test.max())
plt.plot([line_min, line_max], [line_min, line_max], 'k--', lw=2, label='Ideal Fit Line')
# Polynomial Regression best-fit line
poly_coeff = np.polyfit(y_test, poly_y_pred_test, deg=1)  # Fit a linear best-fit line
poly_best_fit_line = np.poly1d(poly_coeff)
plt.plot(np.sort(y_test), poly_best_fit_line(np.sort(y_test)), color='grey', linestyle='--', linewidth=2, label='Polynomial Best Fit Line')
# Setting labels and title with increased font size
plt.xlabel('Y_test (True)', fontsize=20)  # Increase x-axis label size
plt.ylabel('Y_test (Predicted)', fontsize=20)  # Increase y-axis label size
plt.title('Y_test (True) vs Y_test (Predicted): Decision Tree Regressor vs Polynomial Regression', fontsize=19)  # Increase title size

# Enlarging tick labels
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

# Adding legend with enlarged label font size
plt.legend(fontsize=18)  # Adjust legend font size as needed
plt.show()


# (optional)Test set Decision Tree 2D Plot : where the result presented as color scale
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
scatter1= plt.scatter(x1, x2, c=y, cmap='viridis', s=5)
plt.colorbar(scatter1, label='Color Scale')  
plt.title('Original Data')
plt.subplot(1, 2, 2)
scatter2 = plt.scatter(X_test[:, 0], X_test[:, 1], c=decision_tree_y_pred_test, cmap='viridis', s=5)
plt.colorbar(scatter2, label='Color Scale')  
plt.title('Test set: Decision Tree Prediction')
plt.show()
# (optional)Train set Decision Tree 2D Plot : where the result presented as color scale
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
scatter3= plt.scatter(x1, x2, c=y, cmap='viridis', s=5)
plt.colorbar(scatter3, label='Color Scale')  
plt.title('Original Data')
plt.subplot(1, 2, 2)
scatter4 = plt.scatter(X_train[:, 0], X_train[:, 1], c=decision_tree_y_pred_train, cmap='viridis', s=5)
plt.colorbar(scatter4, label='Color Scale')  
plt.title('Trian set: Decision Tree Prediction')
plt.show()






