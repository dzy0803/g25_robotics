#!/usr/bin/env python3

import os  # Import os for directory operations
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

from robot_localization_system import FilterConfiguration, Map, RobotEstimator

# -----------------------------
# Simulator Configuration
# -----------------------------
class SimulatorConfiguration(object):
    def __init__(self):
        self.dt = 0.1
        self.total_time = 1000
        self.time_steps = int(self.total_time / self.dt)

        # Control inputs (linear and angular velocities)
        self.v_c = 1.0  # Linear velocity [m/s]
        self.omega_c = 0.1  # Angular velocity [rad/s]


# -----------------------------
# Controller Placeholder
# -----------------------------
class Controller(object):
    def __init__(self, config):
        self._config = config

    def next_control_input(self, x_est, Sigma_est):
        return [self._config.v_c, self._config.omega_c]


# -----------------------------
# Simulator Implementation
# -----------------------------
class Simulator(object):

    # Initialize
    def __init__(self, sim_config, filter_config, map):
        self._config = sim_config
        self._filter_config = filter_config
        self._map = map

    # Reset the simulator to the start conditions
    def start(self):
        self._time = 0
        # Commented out to start without a randomised initial estimation error for a fair comparison
        # self._x_true = np.random.multivariate_normal(self._filter_config.x0, self._filter_config.Sigma0)
        self._x_true = self._filter_config.x0.copy()  # Start at initial state
        self._u = [0, 0]

    def set_control_input(self, u):
        self._u = u

    # Predict the state forwards to the next timestep
    # edited v to process noise here for easier understanding

    def step(self):
        dt = self._config.dt
        v_c = self._u[0]
        omega_c = self._u[1]
        V = self._filter_config.V

        # Add process noise
        process_noise = np.random.multivariate_normal(mean=[0.0, 0.0, 0.0], cov=V * dt)
        # process_noise = np.random.multivariate_normal(mean=[0.005, 0.005, 0.001], cov=V * dt)

        # Update true state
        self._x_true += np.array([
            v_c * np.cos(self._x_true[2]) * dt,
            v_c * np.sin(self._x_true[2]) * dt,
            omega_c * dt
        ]) + process_noise

        # Normalize the angle
        self._x_true[-1] = np.arctan2(np.sin(self._x_true[-1]),
                                      np.cos(self._x_true[-1]))
        self._time += dt
        return self._time

    # Get the observations to the landmarks.
    def landmark_measurements(self):
        # Generate range and bearing measurements with noise
        return self.generate_measurements(self._x_true)

    def x_true_state(self):
        return self._x_true.copy()

    def generate_measurements(self, true_state):
        """
        Generate range and bearing measurements for each landmark based on the true state.

        :param true_state: The true state of the robot [x, y, theta].
        :return: Tuple of arrays (y_range, y_bearing).
        """
        y_range = []
        y_bearing = []
        for lm in self._map.landmarks:
            dx = lm[0] - true_state[0]
            dy = lm[1] - true_state[1]
            range_meas = np.sqrt(dx**2 + dy**2) + np.random.randn() * np.sqrt(self._filter_config.W_range)
            bearing_meas = (np.arctan2(dy, dx) - true_state[2]) + np.random.randn() * np.sqrt(self._filter_config.W_bearing)
            bearing_meas = np.arctan2(np.sin(bearing_meas), np.cos(bearing_meas))  # Normalize
            y_range.append(range_meas)
            y_bearing.append(bearing_meas)
        return np.array(y_range), np.array(y_bearing)


# -----------------------------
# Helper Function to Wrap Angles
# -----------------------------
def wrap_angle(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


# -----------------------------
# Main Execution
# -----------------------------
def main():
    # -------------------------
    # Define Save Directory
    # -------------------------
    save_dir = 'C:/Users/ziyar/Desktop/UCL/Estimation and Control/Reports/Final Report/activity 4'  # Change this to your desired directory path
    os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist

    # -------------------------
    # Create Configurations
    # -------------------------
    # Create the simulator configuration.
    sim_config = SimulatorConfiguration()

    # Create the filter configuration. If you want
    # to investigate mis-tuning the filter,
    # create a different filter configuration for
    # the simulator and for the filter, and
    # change the parameters between them.
    filter_config = FilterConfiguration()

    # Create the map object for the landmarks.
    map_ = Map()

    # Create the controller. This just provides
    # fixed control inputs for now.
    controller = Controller(sim_config)

    # Create the simulator object and start it.
    simulator = Simulator(sim_config, filter_config, map_)
    simulator.start()

    # Create the estimator and start it.
    estimator = RobotEstimator(filter_config, map_)
    estimator.start()

    # Extract the initial estimates from the filter
    # (which are the initial conditions) and use
    # these to generate the control for the first timestep.
    x_est, Sigma_est = estimator.estimate()
    u = controller.next_control_input(x_est, Sigma_est)

    # Arrays to store data for plotting
    x_true_history = []
    x_est_history = []
    Sigma_est_history = []  # To store full covariance matrices

    # Main loop
    for step in range(sim_config.time_steps):
        # Set the control input and propagate the
        # step the simulator with that control input.
        simulator.set_control_input(u)
        simulation_time = simulator.step()

        # Predict the Kalman filter with the same
        # control inputs to the same time.
        estimator.set_control_input(u)
        estimator.predict_to(simulation_time)

        # Get the landmark observations.
        y_range, y_bearing = simulator.landmark_measurements()

        # Update the filter with the latest observations.
        estimator.update_from_landmark_range_bearing_observations(y_range, y_bearing)
        # estimator.update_from_landmark_range_observations(y_range)

        # Get the current state estimate.
        x_est, Sigma_est = estimator.estimate()

        # Figure out what the controller should do next.
        u = controller.next_control_input(x_est, Sigma_est)

        # Store data for plotting.
        x_true_history.append(simulator.x_true_state())
        x_est_history.append(x_est.copy())
        Sigma_est_history.append(Sigma_est.copy())  # Store full covariance matrices

    # Convert history lists to arrays.
    x_true_history = np.array(x_true_history)
    x_est_history = np.array(x_est_history)
    Sigma_est_history = np.array(Sigma_est_history)

    # Define a global font size for consistency
    GLOBAL_FONT_SIZE = 18  # Adjust as needed for A4 printing

    # -------------------------
    # Plotting: True Path, Estimated Path, Landmarks
    # -------------------------
    plt.figure(figsize=(10, 8))  # Original figure size maintained
    plt.plot(x_true_history[:, 0], x_true_history[:, 1], label='True Path', linewidth=2)
    plt.plot(x_est_history[:, 0], x_est_history[:, 1], label='Estimated Path', linestyle='--', linewidth=2)
    plt.scatter(map_.landmarks[:, 0], map_.landmarks[:, 1],
                marker='x', color='red', label='Landmarks', s=100)
    plt.legend(fontsize=GLOBAL_FONT_SIZE)
    plt.xlabel('X position [m]', fontsize=GLOBAL_FONT_SIZE)
    plt.ylabel('Y position [m]', fontsize=GLOBAL_FONT_SIZE)
    plt.title('Unicycle Robot Localization using Extended Kalman Filter', fontsize=GLOBAL_FONT_SIZE + 4)
    plt.axis('equal')

    # Customize tick parameters for larger font size
    plt.xticks(fontsize=GLOBAL_FONT_SIZE)
    plt.yticks(fontsize=GLOBAL_FONT_SIZE)

    plt.grid(True)
    plt.tight_layout()

    # Save the plot
    plt.savefig(os.path.join(save_dir, 'true_estimated_paths.png'), dpi=300)

    # Display the plot
    plt.show()

    # -------------------------
    # Estimation Error and RMSE
    # -------------------------
    state_name = ['x', 'y', 'θ']
    units = ['m', 'm', 'rad']  # Define units for each state variable

    estimation_error = x_est_history - x_true_history
    estimation_error[:, -1] = wrap_angle(estimation_error[:, -1])

    # Compute RMSE for each state variable over time
    rmse = np.sqrt(np.cumsum(estimation_error**2, axis=0) / (np.arange(1, len(estimation_error) + 1).reshape(-1, 1)))

    # Plot the estimation error, ±2σ bounds, and RMSE for each state variable
    for s in range(3):
        plt.figure(figsize=(12, 6))  # Original figure size maintained

        # Plot Estimation Error
        plt.plot(estimation_error[:, s],
                 label=f'Estimation Error in {state_name[s]}',
                 color='grey',
                 linewidth=2)

        # Plot RMSE
        plt.plot(rmse[:, s],
                 color='orange',
                 linestyle='-',
                 linewidth=2.5,
                 label='RMSE')

        # Plot ±2σ Bounds
        two_sigma = 2 * np.sqrt(Sigma_est_history[:, s, s])
        plt.plot(two_sigma,
                 linestyle='--',
                 color='red',
                 linewidth=2,
                 label='+2σ Bound')
        plt.plot(-two_sigma,
                 linestyle='--',
                 color='green',
                 linewidth=2,
                 label='-2σ Bound')

        # Add horizontal lines at ±0.1 with appropriate units
        if s <= 1:
            plt.axhline(0.1, color='blue', linestyle='-', linewidth=2, label=f'±0.1 {units[s]} Threshold')
            plt.axhline(-0.1, color='blue', linestyle='-', linewidth=2)

        # Set labels with units
        plt.xlabel('Time Step [1/10 s]', fontsize=GLOBAL_FONT_SIZE)
        plt.ylabel(f'Error in {state_name[s]} [{units[s]}]', fontsize=GLOBAL_FONT_SIZE)

        # Set title with state name and units
        plt.title(f'Estimation Error, ±2σ Bounds, and RMSE for {state_name[s]}', fontsize=GLOBAL_FONT_SIZE + 4)

        # Customize tick parameters for larger font size
        plt.xticks(fontsize=GLOBAL_FONT_SIZE)
        plt.yticks(fontsize=GLOBAL_FONT_SIZE)

        # Configure legend to avoid duplicate labels
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), fontsize=GLOBAL_FONT_SIZE, ncol=3)

        # Enable grid and adjust layout
        plt.grid(True)
        plt.tight_layout()

        # Define filename based on state
        filename = f'estimation_error_{state_name[s]}.png'

        # Save the plot
        plt.savefig(os.path.join(save_dir, filename), dpi=300)

        # Display the plot
        plt.show()

    # -------------------------
    # Compute NEES
    # -------------------------
    nees = np.zeros(len(estimation_error))
    for t in range(len(estimation_error)):
        error = estimation_error[t].reshape(-1, 1)  # Shape: (3,1)
        covariance = Sigma_est_history[t]
        try:
            # Used np.linalg.solve for better numerical stability
            nees[t] = error.T @ np.linalg.solve(covariance, error)
        except np.linalg.LinAlgError:
            print(f"Covariance matrix at time step {t} is singular or not invertible.")
            nees[t] = np.nan  # Assign NaN if inversion fails

    # Compute NEES confidence bounds
    alpha = 0.05  # Significance level for 95% confidence interval
    dof = 3       # Degrees of freedom (state dimension)
    lower_bound = chi2.ppf(alpha / 2, dof)
    upper_bound = chi2.ppf(1 - alpha / 2, dof)

    # -------------------------
    # Plot NEES over Time with Confidence Bounds
    # -------------------------
    plt.figure(figsize=(12, 6))  # Original figure size maintained
    plt.plot(nees, label='NEES', color='blue', linewidth=2)
    plt.axhline(lower_bound, color='red', linestyle='--', linewidth=2, label=f'Lower 95% Bound ({lower_bound:.2f})')
    plt.axhline(upper_bound, color='green', linestyle='--', linewidth=2, label=f'Upper 95% Bound ({upper_bound:.2f})')
    plt.axhline(dof, color='black', linestyle='-', linewidth=2, label=f'Optimal NEES Value (n={dof})')

    plt.xlabel('Time Step', fontsize=GLOBAL_FONT_SIZE)
    plt.ylabel('NEES Value', fontsize=GLOBAL_FONT_SIZE)
    plt.xlim(0,1000)
    # ness plotted to time 100 to be inside the landmark trajectory.
    plt.ylim(0, np.max(nees[:1000]))

    # Set title with state name and units
    plt.title('Normalized Estimation Error Squared (NEES) Over Time', fontsize=GLOBAL_FONT_SIZE + 4)

    # Customize tick parameters for larger font size
    plt.xticks(fontsize=GLOBAL_FONT_SIZE)
    plt.yticks(fontsize=GLOBAL_FONT_SIZE)

    plt.legend(fontsize=GLOBAL_FONT_SIZE, ncol=2)
    plt.grid(True)
    plt.tight_layout()

    # Define filename
    filename = 'nees_over_time.png'

    # Save the plot
    plt.savefig(os.path.join(save_dir, filename), dpi=300)

    # Display the plot
    plt.show()


if __name__ == "__main__":
    main()