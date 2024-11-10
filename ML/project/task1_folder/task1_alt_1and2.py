import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import os
import seaborn as sns


# Constants
m = 1.0  # Mass (kg)
b = 10  # Friction coefficient
k_p = 50  # Proportional gain
k_d = 10   # Derivative gain
dt = 0.01  # Time step
num_samples = 1000  # Number of samples in dataset

# Generate synthetic data for trajectory tracking
t = np.linspace(0, 10, num_samples)
q_target = np.sin(t)
dot_q_target = np.cos(t)

# Initial conditions for training data generation
q = 0
dot_q = 0
X = []
Y = []

for i in range(num_samples):
    # PD control output
    tau = k_p * (q_target[i] - q) + k_d * (dot_q_target[i] - dot_q)
    # Ideal motor dynamics (variable mass for realism)
    #m_real = m * (1 + 0.1 * np.random.randn())  # Mass varies by +/-10%
    ddot_q_real = (tau - b * dot_q) / m
    
    # Calculate error
    ddot_q_ideal = (tau) / m
    ddot_q_error = ddot_q_ideal - ddot_q_real
    
    # Store data
    X.append([q, dot_q, q_target[i], dot_q_target[i]])
    Y.append(ddot_q_error)
    
    # Update state
    dot_q += ddot_q_real * dt
    q += dot_q * dt

# Convert data for PyTorch
X_tensor = torch.tensor(X, dtype=torch.float32)
Y_tensor = torch.tensor(Y, dtype=torch.float32).view(-1, 1)

# Dataset and DataLoader
train_dataset = TensorDataset(X_tensor, Y_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  

# MLP Model Definition
class ShallowCorrectorMLP(nn.Module):
    def __init__(self, num_hidden_units=32):
        super(ShallowCorrectorMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(4, num_hidden_units),
            nn.ReLU(),
            nn.Linear(num_hidden_units, 1)
        )

    def forward(self, x):
        return self.layers(x)


# MLP Model Definition
class DeepCorrectorMLP(nn.Module):
    def __init__(self, num_hidden_units=32):
        super(DeepCorrectorMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(4, num_hidden_units),
            nn.ReLU(),
            nn.Linear(num_hidden_units, num_hidden_units),
            nn.ReLU(),
            nn.Linear(num_hidden_units, 1)
        )

    def forward(self, x):
        return self.layers(x)
    
 
# TODO Start Point ---------------------------------------------------------------------------------------------------------- 
# Creating directory for saving results
results_dir = 'task1_1_and_2_results'  
os.makedirs(results_dir, exist_ok=True)

# Experiment parameters: defined for automating the experiment process
hidden_nodes_list = [32, 64, 96, 128]  # change the list to hidden_nodes_list = [32] for task1.4
models = {
    'ShallowCorrectorMLP': ShallowCorrectorMLP,
    'DeepCorrectorMLP': DeepCorrectorMLP
} # define the 'dict' for two models with 'key-value' data structure

# store all trainning losses
train_losses_matrix = []


for model_name, model_class in models.items():
    model_losses = []
    train_losses_each_setting = []  # store for trainning losses with different hidden nodes numbers
    for num_hidden_nodes in hidden_nodes_list:
        print(f'\nTraining {model_name} model with {num_hidden_nodes} hidden nodes......')

        model = model_class(num_hidden_units=num_hidden_nodes)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-4) 
        
        
        # Training Loop
        epochs = 100
        train_losses = []
        
        model.train()
        
        for epoch in range(epochs):
            epoch_loss = 0
            for data, target in train_loader:
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            train_losses.append(epoch_loss / len(train_loader))
            print(f'Hidden Units: {num_hidden_nodes}, Epoch {epoch+1}/{epochs}, Loss: {train_losses[-1]:.6f}')

        model_losses.append(train_losses[-1])
        train_losses_each_setting.append(train_losses) 
        # Testing Phase: Simulate trajectory tracking
        q_test = 0
        dot_q_test = 0
        q_real = []
        q_real_corrected = []

        # integration with only PD Control
        for i in range(len(t)):
                tau = k_p * (q_target[i] - q_test) + k_d * (dot_q_target[i] - dot_q_test)
                ddot_q_real = (tau - b * dot_q_test) / m
                dot_q_test += ddot_q_real * dt
                q_test += dot_q_test * dt
                q_real.append(q_test)

        model.eval()

        q_test = 0
        dot_q_test = 0
        for i in range(len(t)):
                # Apply MLP correction
                tau = k_p * (q_target[i] - q_test) + k_d * (dot_q_target[i] - dot_q_test)
                inputs = torch.tensor([q_test, dot_q_test, q_target[i], dot_q_target[i]], dtype=torch.float32)
                with torch.no_grad():
                    correction = model(inputs.unsqueeze(0)).item()
                ddot_q_corrected =(tau - b * dot_q_test + correction) / m
                dot_q_test += ddot_q_corrected * dt
                q_test += dot_q_test * dt
                q_real_corrected.append(q_test)

       
                
        # Create subdirectory to save figures 
        sub_dir = os.path.join(results_dir, f'{model_name}_{num_hidden_nodes}_hidden_nodes')
        os.makedirs(sub_dir, exist_ok=True)
        
        # Save results for Trajectory Tracking and Position Error with and without MLP Correction
        plt.figure(figsize=(12, 6))
        plt.plot(t, q_target, 'r-', label='Target')
        plt.plot(t, q_real, 'b--', label='PD Only')
        plt.plot(t, q_real_corrected, 'g:', label='PD + MLP Correction')
        plt.title(f'Trajectory Tracking with and without {model_name}_{num_hidden_nodes} Correction', fontsize=22)
        plt.xlabel('Time [s]', fontsize=18)
        plt.ylabel('Position', fontsize=18 )
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.legend(fontsize=16)
        plt.savefig(os.path.join(sub_dir, 'Trajectory_Tracking_Position_compare.png'))
        plt.close()

       
        plt.figure(figsize=(12, 6))
        plt.plot(t, q_target-q_real, 'b--', label='PD Only')
        plt.plot(t, q_target-q_real_corrected, 'g:', label='PD + MLP Correction')
        plt.title(f'Position Error with and without {model_name}_{num_hidden_nodes} Correction', fontsize=22)
        plt.xlabel('Time [s]' , fontsize=18)
        plt.ylabel('Error', fontsize=18)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.legend(fontsize=16)
        plt.savefig(os.path.join(sub_dir, 'Position_Error_compare.png'))
        plt.close()
        
        # Plot and save trainning individual training losses figures
        plt.figure(figsize=(12, 6))
        plt.plot(train_losses, label=f'{model_name}, Hidden Nodes: {num_hidden_nodes}')
        plt.title(f'Training Losses for {model_name} with {num_hidden_nodes} Hidden Nodes', fontsize=22)
        plt.xlabel('Epoch', fontsize=18)
        plt.ylabel('Training Loss', fontsize=18)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.legend(fontsize=16)
        plt.savefig(os.path.join(sub_dir, 'Training_loss_vs_Epoch_plot.png'))
        plt.close()
        

    train_losses_matrix.append(model_losses)
    
    # training losses plosts for different numbers of hidden_units setting
    plt.figure(figsize=(12, 6))
    for i, num_hidden_nodes in enumerate(hidden_nodes_list):
        plt.plot(train_losses_each_setting[i], label=f'Hidden Nodes: {num_hidden_nodes}')
    plt.xlabel('Epoch' ,fontsize=18)
    plt.ylabel('Training Loss',fontsize=18)
    plt.title(f'Training Loss with Different Number Hidden Nodes for {model_name}', fontsize=22)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16)
    plt.savefig(os.path.join(results_dir, f'{model_name}_losses_combinaion.png'))
    plt.show()

# Creating and save heatmap 
plt.figure(figsize=(12, 6))
sns.heatmap(train_losses_matrix, annot=True, fmt=".6f", xticklabels=hidden_nodes_list, yticklabels=models.keys())
plt.title('Heatmap of Training Losses for Different Models with Different Hidden Nodes', fontsize=22)
plt.xlabel('Number of Hidden Units', fontsize=18)
plt.ylabel('Model Type', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(fontsize=16)
plt.savefig(os.path.join(results_dir, 'heatmap.png'))
plt.show()

# End Point --------------------------------------------------------------------------------------------------------------   







