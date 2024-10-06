import numpy as np
import time
import os
import matplotlib.pyplot as plt
from simulation_and_control import pb, MotorCommands, PinWrapper, feedback_lin_ctrl, SinusoidalReference 


def main():
    # Configuration for the simulation
    conf_file_name = "pandaconfig.json"  # Configuration file for the robot
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    sim = pb.SimInterface(conf_file_name, conf_file_path_ext=cur_dir)  # Initialize simulation interface

    # Get active joint names from the simulation
    ext_names = sim.getNameActiveJoints()
    ext_names = np.expand_dims(np.array(ext_names), axis=0)  # Adjust the shape for compatibility

    source_names = ["pybullet"]  # Define the source for dynamic modeling

    # Create a dynamic model of the robot
    dyn_model = PinWrapper(conf_file_name, "pybullet", ext_names, source_names, False, 0, cur_dir)
    num_joints = dyn_model.getNumberofActuatedJoints()

    # Print initial joint angles
    print(f"Initial joint angles: {sim.GetInitMotorAngles()}")

    # Sinusoidal reference
    # Specify different amplitude values for each joint
    amplitudes = [np.pi/4, np.pi/6, np.pi/4, np.pi/4, np.pi/4, np.pi/4, np.pi/4]  # Example amplitudes for joints
    # Specify different frequency values for each joint
    frequencies = [0.4, 0.5, 0.4, 0.4, 0.4, 0.4, 0.4]  # Example frequencies for joints

    # Convert lists to NumPy arrays for easier manipulation in computations
    amplitude = np.array(amplitudes)
    frequency = np.array(frequencies)
    ref = SinusoidalReference(amplitude, frequency, sim.GetInitMotorAngles())  # Initialize the reference
    
    
    # Simulation parameters
    time_step = sim.GetTimeStep()
    current_time = 0
    max_time = 10  # seconds
    
    # Command and control loop
    cmd = MotorCommands()  # Initialize command structure for motors
    # PD controller gains
    kp = 1000
    kd = 100

    # Initialize data storage
    tau_mes_all = []
    regressor_all = []
    
    # Data collection loop
    while current_time < max_time:
        # Measure current state
        q_mes = sim.GetMotorAngles(0)
        qd_mes = sim.GetMotorVelocities(0)
        qdd_mes = sim.ComputeMotorAccelerationTMinusOne(0)
        
        # Compute sinusoidal reference trajectory
        q_d, qd_d = ref.get_values(current_time)  # Desired position and velocity
        
        # Control command
        cmd.tau_cmd = feedback_lin_ctrl(dyn_model, q_mes, qd_mes, q_d, qd_d, kp, kd)
        sim.Step(cmd, "torque")

        # Get measured torque
        tau_mes = sim.GetMotorTorques(0)

        if dyn_model.visualizer: 
            for index in range(len(sim.bot)):  # Conditionally display the robot model
                q = sim.GetMotorAngles(index)
                dyn_model.DisplayModel(q)  # Update the display of the robot model

        # Exit logic with 'q' key
        keys = sim.GetPyBulletClient().getKeyboardEvents()
        qKey = ord('q')
        if qKey in keys and keys[qKey] and sim.GetPyBulletClient().KEY_WAS_TRIGGERED:
            break

        
        # TODO Compute regressor and store it
        cur_regressor = dyn_model.ComputeDynamicRegressor(q_mes, qd_mes, qdd_mes)
        
        regressor_all.append(cur_regressor)
        tau_mes_all.append(tau_mes)
        
        current_time += time_step
        # Optional: print current time
        print(f"Current time in seconds: {current_time:.2f}")

    # TODO After data collection, stack all the regressor and all the torquen and compute the parameters 'a'  using pseudoinverse

    # Stack all the regressor and all the torque
    regressor_all = np.array(regressor_all)
    tau_mes_all = np.array(tau_mes_all)
    X=regressor_all.reshape(70007,70)
    X_trimed = X[7001:,:]
    Y=tau_mes_all.reshape(70007,1)
    Y_trimed = Y[7001:,:]
  
    # Compute the parameters 'a' usg pseudoinverse
    I = np.eye(X_trimed.shape[1])
    a = np.linalg.inv(X_trimed.T @ X_trimed+1*I) @ X_trimed.T @ Y_trimed
    print(a.shape)



    # TODO compute the metrics for the linear model
    Y_pred = X_trimed @ a
    # RSS
    RSS = np.sum((Y_trimed - Y_pred) ** 2)

    # TSS
    TSS = np.sum((Y_trimed - np.mean(Y)) ** 2)

    # R-squared
    r_squared = 1 - (RSS / TSS)

    # Adjusted R-squared
    n = X_trimed.shape[0]  # Number of data points
    p = X_trimed.shape[1]  # Number of predictors
    adj_r_squared = 1 - ((1 - r_squared) * (n - 1)) / (n - p - 1)

    # Mean squared error
    mse = np.mean((Y_trimed - Y_pred) ** 2)

    # F-statistics
    F_stat = (TSS-RSS)/p/(RSS/(n-p-1))

    # Confidence intervals
    se = np.sqrt(np.diagonal(mse * np.linalg.inv(X_trimed.T @ X_trimed+1*I)))
    conf_intervals = [a-1.96*se, a+1.96*se]

    print(f"RSS: {RSS}")
    print(f"TSS: {TSS}")
    print(f"R-squared: {r_squared}")
    print(f"Mean squared Error: {mse}")
    print(f"Adjusted R-squared: {adj_r_squared}")
    print(f"F-statistics: {F_stat}")
    print(f"Confidence intervals: {conf_intervals}")

    plt.figure(figsize=(10, 5))
    plt.scatter(Y_trimed,Y_pred,color='blue',alpha=0.5)
    plt.plot([Y_trimed.min(), Y_trimed.max()], [Y_trimed.min(), Y_trimed.max()], color='red', linestyle='--', label='Perfect Prediction')
    plt.xlabel("Y_trimed")
    plt.ylabel("Y_pred")
    plt.grid()
 


   
    # TODO plot the  torque prediction error for each joint (optional)
    # Joint 1

    joint_1_X = regressor_all[1000:,0,:]
    joint_1_Y = tau_mes_all[1000:,0]
    a_1 = np.linalg.inv(joint_1_X.T @ joint_1_X+1*I) @ joint_1_X.T @ joint_1_Y
    joint_1_Y_pred = joint_1_X @ a_1
    
    plt.figure(figsize=(10, 5))
    plt.plot(joint_1_Y, label='True Torque')
    plt.plot(joint_1_Y_pred, label='Predicted Torque')
    plt.xlabel('Time')
    plt.ylabel('Torque')
    plt.title('Joint 1 Torque Prediction')

    # Joint 2
    joint_2_X = regressor_all[1000:,1,:]
    joint_2_Y = tau_mes_all[1000:,1]
    a_2 = np.linalg.inv(joint_2_X.T @ joint_2_X+1*I) @ joint_2_X.T @ joint_2_Y
    joint_2_Y_pred = joint_2_X @ a_2

    plt.figure(figsize=(10, 5))
    plt.plot(joint_2_Y, label='True Torque')
    plt.plot(joint_2_Y_pred, label='Predicted Torque')
    plt.xlabel('Time')
    plt.ylabel('Torque')
    plt.title('Joint 2 Torque Prediction')

    plt.show()
    plt.show()
    plt.show()

    

if __name__ == '__main__':
    main()
