#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from robot_localization_system import FilterConfiguration, Map, RobotEstimator

# Simulator configuration; this only contains
# stuff relevant for the standalone simulator.
class SimulatorConfiguration(object):
    def __init__(self):
        self.dt = 0.1
        self.total_time = 100  # Reduced for quicker testing
        self.time_steps = int(self.total_time / self.dt)

        # Control inputs (linear and angular velocities)
        self.v_c = 1.0  # Linear velocity [m/s]
        self.omega_c = 0.1  # Angular velocity [rad/s]


# Placeholder for a controller.
class Controller(object):
    def __init__(self, config):
        self._config = config

    def next_control_input(self, x_est, Sigma_est):
        return [self._config.v_c, self._config.omega_c]


# This class implements a simple simulator for the unicycle
# robot seen in the lectures.
class Simulator(object):

    # Initialize
    def __init__(self, sim_config, filter_config, map):
        self._config = sim_config
        self._filter_config = filter_config
        self._map = map

    # Reset the simulator to the start conditions
    def start(self):
        self._time = 0
        self._x_true = self._filter_config.x0.copy()  # Start at initial state
        self._u = [0, 0]

    def set_control_input(self, u):
        self._u = u

    # Predict the state forwards to the next timestep
    def step(self):
        dt = self._config.dt
        v_c = self._u[0]
        omega_c = self._u[1]
        V = self._filter_config.V

        # Add process noise
        process_noise = np.random.multivariate_normal(mean=[0, 0, 0], cov=V * dt)

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


# Helper function to wrap angles
def wrap_angle(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


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
Sigma_est_history = []

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

    # Get the current state estimate.
    x_est, Sigma_est = estimator.estimate()

    # Figure out what the controller should do next.
    u = controller.next_control_input(x_est, Sigma_est)

    # Store data for plotting.
    x_true_history.append(simulator.x_true_state())
    x_est_history.append(x_est.copy())
    Sigma_est_history.append(np.diag(Sigma_est).copy())

# Convert history lists to arrays.
x_true_history = np.array(x_true_history)
x_est_history = np.array(x_est_history)
Sigma_est_history = np.array(Sigma_est_history)

# Plotting the true path, estimated path, and landmarks.
plt.figure(figsize=(10, 8))
plt.plot(x_true_history[:, 0], x_true_history[:, 1], label='True Path', linewidth=2)
plt.plot(x_est_history[:, 0], x_est_history[:, 1], label='Estimated Path', linestyle='--', linewidth=2)
plt.scatter(map_.landmarks[:, 0], map_.landmarks[:, 1],
            marker='x', color='red', label='Landmarks', s=100)
plt.legend(fontsize=12)
plt.xlabel('X position [m]', fontsize=14)
plt.ylabel('Y position [m]', fontsize=14)
plt.title('Unicycle Robot Localization using Extended Kalman Filter (EKF)', fontsize=16)
plt.axis('equal')
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot the 2 standard deviation and error history for each state.
state_name = ['x', 'y', 'θ']
estimation_error = x_est_history - x_true_history
estimation_error[:, -1] = wrap_angle(estimation_error[:, -1])

for s in range(3):
    mean_error = np.mean(estimation_error[:, s])
    plt.figure(figsize=(12, 6))
    plt.plot(estimation_error[:, s], label=f'Estimation Error in {state_name[s]}', color='blue')
    two_sigma = 2 * np.sqrt(Sigma_est_history[:, s])
    plt.plot(two_sigma, linestyle='--', color='red', label='+2σ Bound')
    plt.plot(-two_sigma, linestyle='--', color='green', label='-2σ Bound')
    plt.axhline(mean_error, color='orange', linestyle=':', label=f'Mean Error = {mean_error:.4f}')
    plt.xlabel('Time Step', fontsize=14)
    plt.ylabel(f'Error in {state_name[s]}', fontsize=14)
    plt.title(f'Estimation Error and 2σ Bounds for {state_name[s]}', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.show()