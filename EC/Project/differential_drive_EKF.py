#!/usr/bin/env python3

import numpy as np
import time
import os
import matplotlib.pyplot as plt
from simulation_and_control import pb, MotorCommands, PinWrapper, feedback_lin_ctrl, SinusoidalReference, CartesianDiffKin, differential_drive_controller_adjusting_bearing
from simulation_and_control import differential_drive_regulation_controller, regulation_polar_coordinates, regulation_polar_coordinate_quat, wrap_angle, velocity_to_wheel_angular_velocity
import pinocchio as pin
from scipy.linalg import solve_discrete_are

# Global variables
W_range = 0.5 ** 2  # Measurement noise variance (range measurements)


def landmark_range_observations(base_position, landmarks, W_range):
    y = []
    W = W_range
    for lm in landmarks:
        # True range measurement (with noise)
        dx = lm[0] - base_position[0]
        dy = lm[1] - base_position[1]
        range_pred = np.sqrt(dx**2 + dy**2)
        # Avoid division by zero
        if range_pred == 0:
            continue  # Skip this landmark
        range_meas = range_pred + np.random.normal(0, np.sqrt(W))
        y.append(range_meas)
    y = np.array(y)
    return y


def quaternion2bearing(q_w, q_x, q_y, q_z):
    quat = pin.Quaternion(q_w, q_x, q_y, q_z)
    quat.normalize()  # Ensure the quaternion is normalized

    # Convert quaternion to rotation matrix
    rot_quat = quat.toRotationMatrix()

    # Convert rotation matrix to Euler angles (roll, pitch, yaw)
    base_euler = pin.rpy.matrixToRpy(rot_quat)  # Returns [roll, pitch, yaw]

    # Extract the yaw angle
    bearing_ = base_euler[2]

    return bearing_


def wrap_angle(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def init_simulator(conf_file_name):
    """Initialize simulation and dynamic model."""
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    sim = pb.SimInterface(conf_file_name, conf_file_path_ext=cur_dir)
    
    ext_names = np.expand_dims(np.array(sim.getNameActiveJoints()), axis=0)
    source_names = ["pybullet"]
    
    dyn_model = PinWrapper(conf_file_name, "pybullet", ext_names, source_names, False, 0, cur_dir)
    num_joints = dyn_model.getNumberofActuatedJoints()
    
    return sim, dyn_model, num_joints


class FilterConfiguration(object):
    def __init__(self):
        # Process and measurement noise covariance matrices
        self.V = np.diag([0.1, 0.1, 0.05]) ** 2  # Process noise covariance
        # Measurement noise variances
        self.W_range = 0.5 ** 2
        self.W_bearing = (np.pi * 0.5 / 180.0) ** 2  # Convert degrees to radians

        # Initial conditions for the filter
        self.x0 = np.array([2.0, 3.0, np.pi / 4])
        self.Sigma0 = np.diag([1.0, 1.0, 0.5]) ** 2


class Map(object):
    def __init__(self):
        # Define grid spacing and generate grid landmarks
        grid_spacing = 10
        grid_x = np.arange(-20, 21, grid_spacing) + 2.1  # Shifted by 0.1 to avoid robot's initial position
        grid_y = np.arange(-20, 21, grid_spacing) + 3.1  # Shifted by 0.1 to avoid robot's initial position
        self.landmarks = np.array([(x, y) for x in grid_x for y in grid_y])


class RobotEstimator(object):

    def __init__(self, filter_config, map_):
        # Variables which will be used
        self._config = filter_config
        self._map = map_

    # This method MUST be called to start the filter
    def start(self):
        self._t = 0
        self._set_estimate_to_initial_conditions()

    def set_control_input(self, u):
        self._u = u

    # Predict to the time. The time is fed in to
    # allow for variable prediction intervals.
    def predict_to(self, time):
        # What is the time interval length?
        dt = time - self._t

        # Store the current time
        self._t = time

        # Now predict over a duration dt
        self._predict_over_dt(dt)

    # Return the estimate and its covariance
    def estimate(self):
        return self._x_est, self._Sigma_est

    # This method gets called if there are no observations
    def copy_prediction_to_estimate(self):
        self._x_est = self._x_pred
        self._Sigma_est = self._Sigma_pred

    # This method sets the filter to the initial state
    def _set_estimate_to_initial_conditions(self):
        # Initial estimated state and covariance
        self._x_est = self._config.x0.copy()
        self._Sigma_est = self._config.Sigma0.copy()

    # Predict to the time
    def _predict_over_dt(self, dt):
        v_c = self._u[0]
        omega_c = self._u[1]
        V = self._config.V

        # Predict the new state
        self._x_pred = self._x_est + np.array([
            v_c * np.cos(self._x_est[2]) * dt,
            v_c * np.sin(self._x_est[2]) * dt,
            omega_c * dt
        ])
        self._x_pred[-1] = np.arctan2(np.sin(self._x_pred[-1]),
                                      np.cos(self._x_pred[-1]))

        # Predict the covariance
        A = np.array([
            [1, 0, -v_c * np.sin(self._x_est[2]) * dt],
            [0, 1,  v_c * np.cos(self._x_est[2]) * dt],
            [0, 0, 1]
        ])

        self._kf_predict_covariance(A, V * dt)

    # Predict the EKF covariance
    def _kf_predict_covariance(self, A, V):
        self._Sigma_pred = A @ self._Sigma_est @ A.T + V

    # Implement the Kalman filter update step.
    def _do_kf_update(self, nu, C, W):
        # Kalman Gain
        SigmaXZ = self._Sigma_pred @ C.T
        SigmaZZ = C @ SigmaXZ + W
        K = SigmaXZ @ np.linalg.inv(SigmaZZ)

        # State update
        self._x_est = self._x_pred + K @ nu

        # Covariance update
        self._Sigma_est = (np.eye(len(self._x_est)) - K @ C) @ self._Sigma_pred

    def update_from_landmark_range_observations(self, y_range):

        # Predicted the landmark measurements and build up the observation Jacobian
        y_pred = []
        C = []
        x_pred = self._x_pred
        valid_indices = []  # Keep track of valid measurements
        for idx, lm in enumerate(self._map.landmarks):

            dx_pred = lm[0] - x_pred[0]
            dy_pred = lm[1] - x_pred[1]
            range_pred = np.sqrt(dx_pred**2 + dy_pred**2)

            # Avoid division by zero
            if range_pred == 0:
                continue  # Skip this landmark

            y_pred.append(range_pred)
            valid_indices.append(idx)

            # Jacobian of the measurement model
            C_range = np.array([
                -(dx_pred) / range_pred,
                -(dy_pred) / range_pred,
                0
            ])
            C.append(C_range)
        # Convert lists to arrays
        C = np.array(C)
        y_pred = np.array(y_pred)

        # Adjust y_range to only include valid measurements
        y_range = y_range[valid_indices]

        # Innovation. Look new information! (geddit?)
        nu = y_range - y_pred

        # Since we are observing a subset of landmarks, build the covariance matrix accordingly
        W_landmarks = self._config.W_range * np.eye(len(y_pred))
        self._do_kf_update(nu, C, W_landmarks)

        # Angle wrap afterwards
        self._x_est[-1] = np.arctan2(np.sin(self._x_est[-1]),
                                     np.cos(self._x_est[-1]))


class RegulatorModel:
    def __init__(self, N, q, m, n):
        self.A = None
        self.B = None
        self.C = None
        self.Q = None
        self.R = None
        self.N = N
        self.q = q  # output dimension
        self.m = m  # input dimension
        self.n = n  # state dimension
        # self.P=None

    def compute_terminal_weight_matrix(self):

        self.P = solve_discrete_are(self.A, self.B, self.Q, self.R, balanced=True)

    def compute_H_and_F(self, S_bar, T_bar, Q_bar, R_bar):
        # Compute H
        H = np.dot(S_bar.T, np.dot(Q_bar, S_bar)) + R_bar

        # Compute F
        F = np.dot(S_bar.T, np.dot(Q_bar, T_bar))

        return H, F

    def propagation_model_regulator_fixed_std(self):
        S_bar = np.zeros((self.N * self.q, self.N * self.m))
        T_bar = np.zeros((self.N * self.q, self.n))
        Q_bar = np.zeros((self.N * self.q, self.N * self.q))
        R_bar = np.zeros((self.N * self.m, self.N * self.m))

        for k in range(1, self.N + 1):
            for j in range(1, k + 1):
                # Compute the submatrix to assign and check its shape
                submatrix = np.dot(np.dot(self.C, np.linalg.matrix_power(self.A, j - 1)), self.B)
                # Verify submatrix shape aligns with S_bar target slice
                S_bar[(k - 1) * self.q:k * self.q, (k - j) * self.m:(k - j + 1) * self.m] = submatrix
            T_bar[(k - 1) * self.q:k * self.q, :self.n] = np.dot(self.C, np.linalg.matrix_power(self.A, k))
            # if k == self.N:
            #     Q_bar[(k-1)*self.q:k*self.q, (k-1)*self.q:k*self.q] = self.P
            # else:
            Q_bar[(k - 1) * self.q:k * self.q, (k - 1) * self.q:k * self.q] = self.Q
            R_bar[(k - 1) * self.m:k * self.m, (k - 1) * self.m:k * self.m] = self.R

        return S_bar, T_bar, Q_bar, R_bar

    def updateSystemMatrices(self, sim, cur_x, cur_u):
        """
        Get the system matrices A and B according to the dimensions of the state and control input.

        Parameters:
        cur_x, current state around which to linearize
        cur_u, current control input around which to linearize

        Returns:
        A: State transition matrix
        B: Control input matrix
        """
        # Check if state_x_for_linearization and cur_u_for_linearization are provided
        if cur_x is None or cur_u is None:
            raise ValueError(
                "state_x_for_linearization and cur_u_for_linearization are not specified.\n"
                "Please provide the current state and control input for linearization.\n"
                "Hint: Use the goal state (e.g., zeros) and zero control input at the beginning.\n"
                "Also, ensure that you implement the linearization logic in the updateSystemMatrices function."
            )

        # Initialize A and B matrices
        num_states = self.n
        num_controls = self.m
        num_outputs = self.q
        time_step = sim.GetTimeStep()
        # print("Time Step:", time_step)

        # Get the current state and control input
        theta_0 = cur_x[2]
        v_0 = cur_u[0]

        # To avoid singularities, ensure v_0 is not zero
        if v_0 == 0:
            v_0 = 1e-5  # Small non-zero value

        # Compute A_c matrix (partial derivatives of f with respect to x)
        A_c = np.array([
            [0, 0, -v_0 * np.sin(theta_0)],
            [0, 0, v_0 * np.cos(theta_0)],
            [0, 0, 0]
        ])

        # Compute B_c matrix (partial derivatives of f with respect to u)
        B_c = np.array([
            [np.cos(theta_0), 0],
            [np.sin(theta_0), 0],
            [0, 1]
        ])

        # Discretize A and B matrices
        I = np.eye(len(cur_x))
        A = I + time_step * A_c
        B = time_step * B_c

        self.A = A
        self.B = B
        self.C = np.eye(num_states)

        # Number of states
        n = A.shape[0]

        # Remove the controllability rank check
        # The check is not necessary at each time step and can cause issues if the system is marginally controllable
        # If required, you can perform this check outside of this method during initialization

    def setCostMatrices(self, Qcoeff, Rcoeff):
        """
        Get the cost matrices Q and R for the MPC controller.

        Returns:
        Q: State cost matrix
        R: Control input cost matrix
        """
        num_states = self.n
        num_controls = self.m

        # Process Qcoeff
        if np.isscalar(Qcoeff):
            # If Qcoeff is a scalar, create an identity matrix scaled by Qcoeff
            Q = Qcoeff * np.eye(num_states)
        else:
            # Convert Qcoeff to a numpy array
            Qcoeff = np.array(Qcoeff)
            if Qcoeff.ndim != 1 or len(Qcoeff) != num_states:
                raise ValueError(f"Qcoeff must be a scalar or a 1D array of length {num_states}")
            # Create a diagonal matrix with Qcoeff as the diagonal elements
            Q = np.diag(Qcoeff)

        # Process Rcoeff
        if np.isscalar(Rcoeff):
            # If Rcoeff is a scalar, create an identity matrix scaled by Rcoeff
            R = Rcoeff * np.eye(num_controls)
        else:
            # Convert Rcoeff to a numpy array
            Rcoeff = np.array(Rcoeff)
            if Rcoeff.ndim != 1 or len(Rcoeff) != num_controls:
                raise ValueError(f"Rcoeff must be a scalar or a 1D array of length {num_controls}")
            # Create a diagonal matrix with Rcoeff as the diagonal elements
            R = np.diag(Rcoeff)

        # Assign the matrices to the object's attributes
        self.Q = Q
        self.R = R


def main():
    # Configuration for the simulation
    conf_file_name = "robotnik.json"  # Configuration file for the robot
    sim, dyn_model, num_joints = init_simulator(conf_file_name)

    # Adjusting floor friction
    floor_friction = 100
    sim.SetFloorFriction(floor_friction)
    # Getting time step
    time_step = sim.GetTimeStep()
    current_time = 0

    # Initialize data storage
    base_pos_all, base_bearing_all = [], []
    x_est_all = []  # Store estimated states
    Sigma_est_all = []  # Store estimated covariances

    # Initializing MPC
    # Define the matrices
    num_states = 3
    num_controls = 2

    # Horizon length
    N_mpc = 10

    # Initialize the regulator model
    regulator = RegulatorModel(N_mpc, num_states, num_controls, num_states)

    # Define the cost matrices
    Qcoeff = np.array([310, 310, 80])
    Rcoeff = 0.5
    regulator.setCostMatrices(Qcoeff, Rcoeff)

    # Initialize control input to small non-zero values
    u_mpc = [1e-5, 1e-5]

    # Robot parameters
    wheel_radius = 0.11
    wheel_base_width = 0.46

    # MPC control action
    cmd = MotorCommands()  # Initialize command structure for motors
    init_angular_wheels_velocity_cmd = np.array([0.0, 0.0, 0.0, 0.0])
    init_interface_all_wheels = ["velocity", "velocity", "velocity", "velocity"]
    cmd.SetControlCmd(init_angular_wheels_velocity_cmd, init_interface_all_wheels)

    # Initialize EKF components
    filter_config = FilterConfiguration()
    map_ = Map()
    estimator = RobotEstimator(filter_config, map_)
    estimator.start()

    while current_time < 20:
        # Advance simulation
        sim.Step(cmd, "torque")
        time_step = sim.GetTimeStep()
        current_time += time_step

        # Kalman filter prediction
        estimator.set_control_input(u_mpc)
        estimator.predict_to(current_time)

        # Get the measurements from the simulator
        # Measurements of the robot without noise (just for comparison purpose)
        base_pos_no_noise = sim.bot[0].base_position
        base_ori_no_noise = sim.bot[0].base_orientation
        base_bearing_no_noise_ = quaternion2bearing(base_ori_no_noise[3], base_ori_no_noise[0], base_ori_no_noise[1], base_ori_no_noise[2])
        base_lin_vel_no_noise = sim.bot[0].base_lin_vel
        base_ang_vel_no_noise = sim.bot[0].base_ang_vel

        # Measurements of the current state (real measurements with noise)
        base_pos = sim.GetBasePosition()
        base_ori = sim.GetBaseOrientation()
        base_bearing_ = quaternion2bearing(base_ori[3], base_ori[0], base_ori[1], base_ori[2])

        # Get landmark range measurements
        y_range = landmark_range_observations(base_pos, map_.landmarks, W_range)

        # Update the filter with the latest observations
        if len(y_range) > 0:
            estimator.update_from_landmark_range_observations(y_range)
        else:
            estimator.copy_prediction_to_estimate()

        # Get the current state estimate
        x_est, Sigma_est = estimator.estimate()

        # Figure out what the controller should do next
        # Compute the matrices needed for MPC optimization
        # Update the matrices A and B at each time step
        cur_state_x_for_linearization = x_est  # Use the estimated state
        cur_u_for_linearization = u_mpc
        regulator.updateSystemMatrices(sim, cur_state_x_for_linearization, cur_u_for_linearization)

        S_bar, T_bar, Q_bar, R_bar = regulator.propagation_model_regulator_fixed_std()
        H, F = regulator.compute_H_and_F(S_bar, T_bar, Q_bar, R_bar)
        x0_mpc = x_est.flatten()
        # Compute the optimal control sequence
        H_inv = np.linalg.inv(H)
        u_mpc = -H_inv @ F @ x0_mpc
        # Return the optimal control sequence
        u_mpc = u_mpc[0:num_controls]
        # Prepare control command to send to the low level controller
        left_wheel_velocity, right_wheel_velocity = velocity_to_wheel_angular_velocity(u_mpc[0], u_mpc[1], wheel_base_width, wheel_radius)
        angular_wheels_velocity_cmd = np.array([right_wheel_velocity, left_wheel_velocity, left_wheel_velocity, right_wheel_velocity])
        interface_all_wheels = ["velocity", "velocity", "velocity", "velocity"]
        cmd.SetControlCmd(angular_wheels_velocity_cmd, interface_all_wheels)

        # Exit logic with 'q' key (unchanged)
        keys = sim.GetPyBulletClient().getKeyboardEvents()
        qKey = ord('q')
        if qKey in keys and keys[qKey] and sim.GetPyBulletClient().KEY_WAS_TRIGGERED:
            break

        # Store data for plotting if necessary
        base_pos_all.append(base_pos_no_noise)
        base_bearing_all.append(base_bearing_no_noise_)
        x_est_all.append(x_est)
        Sigma_est_all.append(Sigma_est)

    # Convert lists to arrays
    base_pos_all = np.array(base_pos_all)
    base_bearing_all = np.array(base_bearing_all)
    x_est_all = np.array(x_est_all)
    Sigma_est_all = np.array(Sigma_est_all)

    # Plotting
    # Plot true trajectory and estimated trajectory
    plt.figure(figsize=(12, 6))
    plt.plot(base_pos_all[:, 0], base_pos_all[:, 1], label='True Trajectory')
    plt.plot(x_est_all[:, 0], x_est_all[:, 1], label='Estimated Trajectory')
    plt.scatter(base_pos_all[0, 0], base_pos_all[0, 1], c='r', label='Start')
    plt.scatter(base_pos_all[-1, 0], base_pos_all[-1, 1], c='g', label='End')
    plt.scatter(0, 0, c='b', label='Goal')
    plt.legend(fontsize=16)
    plt.xlabel('x', fontsize=18)
    plt.ylabel('y', fontsize=18)
    plt.title('Robot Trajectory', fontsize=22)
    plt.grid()
    plt.show()

    # Plot state estimates vs true states over time
    plt.figure(figsize=(12, 6))
    plt.plot(base_pos_all[:, 0], label='True x')
    plt.plot(x_est_all[:, 0], label='Estimated x')
    plt.plot(base_pos_all[:, 1], label='True y')
    plt.plot(x_est_all[:, 1], label='Estimated y')
    plt.plot(base_bearing_all, label='True theta')
    plt.plot(x_est_all[:, 2], label='Estimated theta')
    plt.axhline(0, color='red', linestyle='--', label='Goal State')
    plt.xlabel('Time Step', fontsize=18)
    plt.ylabel('State', fontsize=18)
    plt.title('State Estimates vs True States', fontsize=22)
    plt.grid()
    plt.legend(fontsize=16)
    plt.show()

    # Plot estimation errors
    plt.figure(figsize=(12, 6))
    error_x = base_pos_all[:, 0] - x_est_all[:, 0]
    error_y = base_pos_all[:, 1] - x_est_all[:, 1]
    error_theta = base_bearing_all - x_est_all[:, 2]
    error_theta = np.arctan2(np.sin(error_theta), np.cos(error_theta))  # Normalize angle error
    plt.plot(error_x, label='Error in x')
    plt.plot(error_y, label='Error in y')
    plt.plot(error_theta, label='Error in theta')
    plt.xlabel('Time Step', fontsize=18)
    plt.ylabel('Estimation Error', fontsize=18)
    plt.title('Estimation Errors over Time', fontsize=22)
    plt.grid()
    plt.legend(fontsize=16)
    plt.show()


if __name__ == '__main__':
    main()
