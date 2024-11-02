#!/usr/bin/env python3

import numpy as np
import scipy.linalg  

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
        grid_x = np.arange(-20, 21, grid_spacing) + 2 # from -20 to 20 inclusive
        grid_y = np.arange(-20, 21, grid_spacing) + 3 # from -20 to 20 inclusive
        self.landmarks = np.array([(x, y) for x in grid_x for y in grid_y])

        # grid_spacing = 1
        # grid_x = np.arange(-2, 3, grid_spacing)+ 2  # from -20 to 20 inclusive
        # grid_y = np.arange(-2, 3, grid_spacing)+ 3   # from -20 to 20 inclusive
                # self.landmarks = np.array([(x, y) for x in grid_x for y in grid_y])

        # # Generate grid landmarks
        # self.landmarks = np.array([[5, 10], [15, 5], [10, 15]])


class RobotEstimator(object):

    def __init__(self, filter_config, map):
        # Variables which will be used
        self._config = filter_config
        self._map = map

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

        self._kf_predict_covariance(A, self._config.V * dt)

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

    def update_from_landmark_range_bearing_observations(self, y_range, y_bearing):
        # Initialize lists for predicted measurements and Jacobians
        y_pred = []
        C = []
        x_pred = self._x_pred

        num_landmarks = len(self._map.landmarks)

        for i, lm in enumerate(self._map.landmarks):
            dx_pred = lm[0] - x_pred[0]
            dy_pred = lm[1] - x_pred[1]
            range_pred = np.sqrt(dx_pred**2 + dy_pred**2)
            bearing_pred = np.arctan2(dy_pred, dx_pred) - x_pred[2]
            bearing_pred = np.arctan2(np.sin(bearing_pred), np.cos(bearing_pred))  # Normalize

            # Append predicted measurements
            y_pred.extend([range_pred, bearing_pred])

            # Jacobian of the range measurement
            C_range = np.array([
                -dx_pred / range_pred,
                -dy_pred / range_pred,
                0
            ])

            # Jacobian of the bearing measurement
            C_bearing = np.array([
                dy_pred / (range_pred**2),
                -dx_pred / (range_pred**2),
                -1
            ])

            # Append Jacobians
            C.append(C_range)
            C.append(C_bearing)

        # Convert lists to arrays
        C = np.array(C)
        y_pred = np.array(y_pred)

        # Construct the measurement vector y by interleaving y_range and y_bearing
        y = np.vstack((y_range, y_bearing)).reshape(-1, order='F')  # Stack measurements

        # Innovation
        nu = y - y_pred

        # Normalize the bearing components in nu
        for i in range(num_landmarks):
            nu[2 * i + 1] = np.arctan2(np.sin(nu[2 * i + 1]), np.cos(nu[2 * i + 1]))

        # Construct the measurement noise covariance matrix as block-diagonal
        W_blocks = []
        for _ in range(num_landmarks):
            Wi = np.array([
                [self._config.W_range, 0],
                [0, self._config.W_bearing]
            ])
            W_blocks.append(Wi)

        W = scipy.linalg.block_diag(*W_blocks)

        # Perform the Kalman filter update
        self._do_kf_update(nu, C, W)

        # Angle wrap afterwards
        self._x_est[-1] = np.arctan2(np.sin(self._x_est[-1]),
                                     np.cos(self._x_est[-1]))
        
    def update_from_landmark_range_observations(self, y_range):

        # Predicted the landmark measurements and build up the observation Jacobian
        y_pred = []
        C = []
        x_pred = self._x_pred
        for lm in self._map.landmarks:

            dx_pred = lm[0] - x_pred[0]
            dy_pred = lm[1] - x_pred[1]
            range_pred = np.sqrt(dx_pred**2 + dy_pred**2)
            y_pred.append(range_pred)

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

        # Innovation. Look new information! (geddit?)
        nu = y_range - y_pred

        # Since we are oberving a bunch of landmarks
        # build the covariance matrix. Note you could
        # swap this to just calling the ekf update call
        # multiple times, once for each observation,
        # as well
        W_landmarks = self._config.W_range * np.eye(len(self._map.landmarks))
        self._do_kf_update(nu, C, W_landmarks)

        # Angle wrap afterwards
        self._x_est[-1] = np.arctan2(np.sin(self._x_est[-1]),
                                     np.cos(self._x_est[-1]))


