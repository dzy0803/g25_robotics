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
    b = np.linalg.pinv(X_trimed.T @ X_trimed) @ X_trimed.T @ Y_trimed
    print(b.shape)
    X_pinv = np.linalg.pinv(X_trimed)
    a = X_pinv @ Y_trimed
    print(a.shape)



    # TODO compute the metrics for the linear model
    Y_pred = X_trimed @ a
    # RSS
    RSS = np.sum((Y_trimed - Y_pred) ** 2)

    # TSS
    TSS = np.sum((Y_trimed - np.mean(Y_trimed)) ** 2)

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
    I = np.eye(X_trimed.shape[1])
    se = np.sqrt(np.diagonal(mse*np.linalg.pinv(X_trimed.T@X_trimed)))
    print(se.shape)
    # conf_intervals = [a-1.96*se, a+1.96*se]

    print(f"RSS: {RSS}")
    print(f"TSS: {TSS}")
    print(f"R-squared: {r_squared}")
    print(f"Mean squared Error: {mse}")
    print(f"Adjusted R-squared: {adj_r_squared}")
    print(f"F-statistics: {F_stat}")
   

    plt.figure(figsize=(10, 5))
    plt.scatter(Y_trimed, Y_pred, color='blue', alpha=0.5)
    plt.plot([Y_trimed.min(), Y_trimed.max()], [Y_trimed.min(), Y_trimed.max()], color='red', linestyle='--', label='Perfect Prediction')
    plt.xlabel("Y_True ")
    plt.ylabel("Y_Pred ")
    plt.title("Predicted Torque against True Torque (Noise=0.0)")
    plt.legend()
    plt.grid()
    plt.show()
   

   
    # TODO plot the  torque prediction error for each joint (optional)

    # Joint 1

    joint_1_X = regressor_all[1001:,0,:]
    joint_1_Y = tau_mes_all[1001:,0]
    a_1 = np.linalg.pinv(joint_1_X.T @ joint_1_X) @ joint_1_X.T @ joint_1_Y
    joint_1_Y_pred = joint_1_X @ a_1
    #mse_1 = np.mean((joint_1_Y - joint_1_Y_pred) ** 2)
    #se_1 = np.sqrt(np.diagonal(mse_1*np.linalg.pinv(np.dot(joint_1_X.T,joint_1_X))))
    #conf_intervals_1 = [a_1-1.96*se_1, a_1+1.96*se_1]

    plt.figure(figsize=(10, 5))
    plt.plot(joint_1_Y, label='True Torque')
    plt.plot(joint_1_Y_pred, label='Predicted Torque')
    # plt.fill_between(len(joint_1_Y), a_1-1.96*se_1, a_1+1.96*se_1, color='gray', alpha=0.3, label='95% Confidence Interval')
    plt.xlabel('Time')
    plt.ylabel('Torque')
    plt.title('Joint 1 Torque Prediction')
    plt.legend()

    # Joint 2
    joint_2_X = regressor_all[1000:,1,:]
    joint_2_Y = tau_mes_all[1000:,1]
    a_2 = np.linalg.pinv(joint_2_X.T @ joint_2_X) @ joint_2_X.T @ joint_2_Y
    joint_2_Y_pred = joint_2_X @ a_2
    # mse_2 = np.mean((joint_2_Y - joint_2_Y_pred) ** 2)
    # se_2 = np.sqrt(np.diagonal(mse*np.linalg.pinv(np.dot(joint_2_X.T,joint_2_X))))
    # conf_intervals_2 = [a_2-1.96*se_2, a_2+1.96*se_2]

    plt.figure(figsize=(10, 5))
    plt.plot(joint_2_Y, label='True Torque')
    plt.plot(joint_2_Y_pred, label='Predicted Torque')
    # plt.fill_between(len(joint_2_Y), a_2-1.96*se_2, a_2+1.96*se_2, color='gray', alpha=0.3, label='95% Confidence Interval')
    plt.xlabel('Time')
    plt.ylabel('Torque')
    plt.title('Joint 2 Torque Prediction')
    plt.legend()

    # Joint 3
    joint_3_X = regressor_all[1001:,2,:]
    joint_3_Y = tau_mes_all[1001:,2]
    a_3= np.linalg.pinv(joint_3_X.T @ joint_3_X) @ joint_3_X.T @ joint_3_Y
    joint_3_Y_pred = joint_3_X @ a_3
    # mse_3 = np.mean((joint_3_Y - joint_3_Y_pred) ** 2)
    # se_3 = np.sqrt(np.diagonal(mse_3*np.linalg.pinv(np.dot(joint_3_X.T,joint_3_X))))
    # conf_intervals = [a_3-1.96*se_3, a_3+1.96*se_3]
    
    plt.figure(figsize=(10, 5))
    plt.plot(joint_3_Y, label='True Torque')
    plt.plot(joint_3_Y_pred, label='Predicted Torque')
    plt.xlabel('Time')
    plt.ylabel('Torque')
    plt.title('Joint 3 Torque Prediction')
    plt.legend()

    # Joint 4
    joint_4_X = regressor_all[1001:,3,:]
    joint_4_Y = tau_mes_all[1001:,3]
    a_4= np.linalg.pinv(joint_4_X.T @ joint_4_X) @ joint_4_X.T @ joint_4_Y
    joint_4_Y_pred = joint_4_X @ a_4
    # mse_4 = np.mean((joint_4_Y - joint_4_Y_pred) ** 2)
    # se_4 = np.sqrt(np.diagonal(mse_4*np.linalg.pinv(np.dot(joint_4_X.T,joint_4_X))))
    # conf_intervals_4 = [a_4-1.96*se_4, a_4+1.96*se_4]
    
    plt.figure(figsize=(10, 5))
    plt.plot(joint_4_Y, label='True Torque')
    plt.plot(joint_4_Y_pred, label='Predicted Torque')
    plt.xlabel('Time')
    plt.ylabel('Torque')
    plt.title('Joint 4 Torque Prediction')
    plt.legend()

    # Joint 5
    joint_5_X = regressor_all[1001:,4,:]
    joint_5_Y = tau_mes_all[1001:,4]
    a_5 = np.linalg.pinv(joint_5_X.T @ joint_5_X) @ joint_5_X.T @ joint_5_Y
    joint_5_Y_pred = joint_5_X @ a_5
    # mse_5 = np.mean((joint_5_Y - joint_5_Y_pred) ** 2)
    # se_5 = np.sqrt(np.diagonal(mse_5*np.linalg.pinv(np.dot(joint_5_X.T,joint_5_X))))
    # conf_intervals_5 = [a_5-1.96*se_5, a_5+1.96*se_5]
    
    plt.figure(figsize=(10, 5))
    plt.plot(joint_5_Y, label='True Torque')
    plt.plot(joint_5_Y_pred, label='Predicted Torque')
    plt.xlabel('Time')
    plt.ylabel('Torque')
    plt.title('Joint 5 Torque Prediction')
    plt.legend()


    # Joint 6
    joint_6_X = regressor_all[1001:,5,:]
    joint_6_Y = tau_mes_all[1001:,5]
    a_6 = np.linalg.pinv(joint_6_X.T @ joint_6_X) @ joint_6_X.T @ joint_6_Y
    joint_6_Y_pred = joint_6_X @ a_6
    # mse_6 = np.mean((joint_6_Y - joint_6_Y_pred) ** 2)
    # se_6 = np.sqrt(np.diagonal(mse_6*np.linalg.pinv(np.dot(joint_6_X.T,joint_6_X))))
    # conf_intervals_6 = [a_6-1.96*se_6, a_6+1.96*se_6]
    
    plt.figure(figsize=(10, 5))
    plt.plot(joint_6_Y, label='True Torque')
    plt.plot(joint_6_Y_pred, label='Predicted Torque')
    plt.xlabel('Time')
    plt.ylabel('Torque')
    plt.title('Joint 6 Torque Prediction')
    plt.legend()



    # Joint 7
    joint_7_X = regressor_all[1001:,6,:]
    joint_7_Y = tau_mes_all[1001:,6]
    a_7 = np.linalg.pinv(joint_7_X.T @ joint_7_X) @ joint_7_X.T @ joint_7_Y
    joint_7_Y_pred = joint_7_X @ a_7
    # mse_7 = np.mean((joint_7_Y - joint_7_Y_pred) ** 2)
    # se_7 = np.sqrt(np.diagonal(mse_7*np.linalg.pinv(np.dot(joint_7_X.T,joint_7_X))))
    # conf_intervals_7 = [a_7-1.96*se_7, a_7+1.96*se_7]
    
    plt.figure(figsize=(10, 5))
    plt.plot(joint_7_Y, label='True Torque')
    plt.plot(joint_7_Y_pred, label='Predicted Torque')
    plt.xlabel('Time')
    plt.ylabel('Torque')
    plt.title('Joint 7 Torque Prediction')
    plt.legend()

    plt.show()

    

if __name__ == '__main__':
    main()
