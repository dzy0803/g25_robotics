
import numpy as np
import os
from matplotlib import pyplot as plt
from simulation_and_control import pb, MotorCommands, PinWrapper, feedback_lin_ctrl, SinusoidalReference, CartesianDiffKin
from skopt import gp_minimize
from skopt.space import Real
from skopt.learning import GaussianProcessRegressor
from skopt.learning.gaussian_process.kernels import RBF
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting
from sklearn.preprocessing import StandardScaler


def fit_gp_model_1d(X_values, y_values):
    # Define the kernel
    rbf_kernel = RBF(
    length_scale=2.0,            # Initial length scale
    length_scale_bounds=(1e-2, 1e2)  # Bounds for length scale
    )
    
    # Create and fit the GP regressor
    gp = GaussianProcessRegressor(
        kernel=rbf_kernel,
        normalize_y=True,
        n_restarts_optimizer=10,
        random_state=42
    )
    gp.fit(X_values, y_values)
    
    return gp

def create_prediction_range_kp0():
    kp0_range = np.linspace(0.1, 1000, 200).reshape(-1, 1)
    return kp0_range

def create_prediction_range_kd0():
    kd0_range = np.linspace(0.0, 100, 200).reshape(-1, 1)
    return kd0_range

def plot_gp_results_1d(kp0_values, kd0_values, tracking_errors, gp_kp0, gp_kd0, tracking_error):
    # Create prediction ranges
    kp0_pred = create_prediction_range_kp0()
    kd0_pred = create_prediction_range_kd0()
    
    # Predict for kp0
    y_mean_kp0, y_std_kp0 = gp_kp0.predict(kp0_pred, return_std=True)
    
    # Predict for kd0
    y_mean_kd0, y_std_kd0 = gp_kd0.predict(kd0_pred, return_std=True)
    
    # Find the index of the minimum cost for Kp and Kd
    min_kp_index = np.argmin(y_mean_kp0)  # Index where Kp cost is minimized
    min_kd_index = np.argmin(y_mean_kd0)  # Index where Kd cost is minimized
    
    # Extract the corresponding optimal values
    optimal_kp_value = kp0_pred[min_kp_index][0]  # Optimal Kp
    optimal_kd_value = kd0_pred[min_kd_index][0]  # Optimal Kd
    
    # Plotting
    plt.figure(figsize=(14, 6))
    
    # First subplot: Kp0 vs Cost Function
    plt.subplot(1, 2, 1)
    plt.plot(kp0_pred, y_mean_kp0, 'k-', lw=1.5, zorder=9, label='Mean prediction')
    plt.fill_between(kp0_pred.ravel(), y_mean_kp0 - 1.96 * y_std_kp0, y_mean_kp0 + 1.96 * y_std_kp0,
                     alpha=0.7, fc='orange', ec='None', label='95% confidence interval')
    
    plt.title("Gaussian process regression on noise-free dataset", fontsize=14)
    plt.xlabel('Kp Value', fontsize=12)
    plt.ylabel('Cost Function', fontsize=12)
    plt.legend(loc='upper left', fontsize=10)
    
    # Add tracking error and minimum Kp value as text inside the graph
    plt.text(0.65, 0.98, f'Tracking Error: {tracking_error:.2f}', transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))
    plt.text(0.65, 0.93, f'Optimal Kp Value: {optimal_kp_value:.2f}', transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))

    # Second subplot: Kd0 vs Cost Function
    plt.subplot(1, 2, 2)
    plt.plot(kd0_pred, y_mean_kd0, 'k-', lw=1.5, zorder=9, label='Mean prediction')
    plt.fill_between(kd0_pred.ravel(), y_mean_kd0 - 1.96 * y_std_kd0, y_mean_kd0 + 1.96 * y_std_kd0,
                     alpha=0.7, fc='orange', ec='None', label='95% confidence interval')

    plt.title("Gaussian process regression on noise-free dataset", fontsize=14)
    plt.xlabel('Kd Value', fontsize=12)
    plt.ylabel('Cost Function', fontsize=12)
    plt.legend(loc='upper left', fontsize=10)
    
    # Add tracking error and minimum Kd value as text inside the graph
    plt.text(0.65, 0.98, f'Tracking Error: {tracking_error:.2f}', transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))
    plt.text(0.65, 0.93, f'Optimal Kd Value: {optimal_kd_value:.2f}', transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))

    plt.tight_layout(pad=2.0)  # Adjust padding to ensure layout is spaced well
    plt.show()
