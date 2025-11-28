#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import torch
import copy

from torch.profiler import profile, record_function, ProfilerActivity
from torch.utils.data import Dataset, DataLoader
# from numba import cuda


# In[2]:


device = torch.device('cuda:0')
print(device)


# In[3]:


d = 3 # dimension of the problem, either 2 or 3, this is d_0
hn = 4 # number of hidden layers, do not change this value
num_samples = 128 * 128


# In[4]:


# function in dim 1
def n_g_square(z): 
    return z **2.0
    
def n_g_trig(z):
    return torch.sin(3*z*torch.pi) # argmax in 1/6 and 5/6

def my_log(z):
    return torch.log(z)/(torch.log(z) - 1.0)

def n_g_log(z):
    return z.detach().apply_(lambda x : 1.0 if x ==0 else torch.log(torch.tensor(x))/(torch.log(torch.tensor(x)) - 1.0))

def chv(z):
    return 2*z - 1.0

def loc_max(z):
    return (25*((chv(z))**3)*(1.0 - 4.0*(chv(z))**2)*torch.exp(-5.0*((chv(z))**2)))/0.8143


# In[5]:


# functions to test in dimension d > 1

def n_g_dim(z, d = d):
    return (1/d) * torch.sum(z**2, dim=1)

def my_log_dim(z, d = d):
    return (1/d) * torch.sum(my_log(z), dim=1)

def n_g_trig_dim(z, d = d):
    return (1/d) * torch.sum(torch.abs(torch.sin(2*z*torch.pi)), dim=1)

def loc_max_dim(z, d = d):
    return (1/d) * torch.sum(loc_max(z), dim=1)


# In[6]:


# NN with leaky relu as activation with two hidden layer
class LeakyNN_weight(nn.Module):
    def __init__(self, input_size, n_hidden1, n_hidden2, output_size):
        super(LeakyNN_weight, self).__init__()
        self.fc1 = nn.Linear(input_size, n_hidden1)
        self.fc2 = nn.Linear(n_hidden1, n_hidden2)
        self.fc3 = nn.Linear(n_hidden2, n_hidden2)
        self.fc4 = nn.Linear(n_hidden2, output_size)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = self.fc4(x)
        
        return x


# In[7]:


# Modular version of LeakyNN_weight, parametrized by L
class LeakyNN(nn.Module):
    def __init__(self, L = 4, input_size = d, n_hidden = 4, output_size = 1):
        """
        L: number of layers, number of hidden layers = L-2
        input_size: dimension of input, here equal to 2 or 3
        n_hidden: width of each hidden layer
        output_size: dimension of output, here equal to 1
        """
        super().__init__()  # Call the parent class's constructor directly
        
        # Ensure that the number of parameters does not change based on L
        assert L >= 2, "L should be at least 2"
        
        # Initialize layers list
        layers = []
        
        # First hidden layer (input_size to n_hidden)
        layers.append(nn.Linear(input_size, n_hidden))
        
        # Add L-2 hidden layers (n_hidden to n_hidden)
        for _ in range(L - 2):
            layers.append(nn.Linear(n_hidden, n_hidden))
        
        # Output layer (n_hidden to output_size)
        layers.append(nn.Linear(n_hidden, output_size))
        
        # Register the layers as ModuleList
        self.layers = nn.ModuleList(layers)
    
    def forward(self, x):
        # Apply LeakyReLU activation after each hidden layer
        for i in range(len(self.layers) - 1):  # Skip the last layer for activation
            x = F.leaky_relu(self.layers[i](x))
        
        # No activation function on the output layer
        x = self.layers[-1](x)
        return x


# In[8]:


L = 4 # To be changed, once you start question 2


# In[9]:


#loss function
loss_func = nn.MSELoss(reduction = 'mean')

# data set for training and validation
def data_tr_val(sigO = torch.tensor(0.1, device = device), batch_size = 128, nbBatch = 128, nbBatchVal = 16, d = d):
    """
    Generates random training and validation datasets with added Gaussian noise.

    Parameters:
        sigO (torch.tensor): Standard deviation of the noise to be added.
        batch_size (int): Number of samples per batch.
        nbBatch (int): Number of batches for the training set.
        nbBatchVal (int): Number of batches for the validation set.
        d (int): Dimension of the input data.

    Returns:
        dataT (DataLoader): PyTorch DataLoader for the training set.
        dataV (DataLoader): PyTorch DataLoader for the validation set.
    """
    # Generate random input data for training (uniformly distributed in [0,1])
    U_train = torch.rand((nbBatch*batch_size, d), device = device)
    #U_train = torch.rand((nbBatch*batch_size, d)).view(-1,1)

    # Generate random input data for validation (same as training)
    U_val = torch.rand((nbBatchVal*batch_size, d), device = device)

    # Gaussian noise for training and validation 
    noise_train = torch.mul(torch.randn(nbBatch*batch_size, device = device), sigO).view(-1,1)
    noise_val = torch.mul(torch.randn(nbBatchVal*batch_size, device = device), sigO).view(-1,1)

    dataT = torch.utils.data.DataLoader(list(zip(U_train, noise_train)), batch_size=batch_size)
    dataV = torch.utils.data.DataLoader(list(zip(U_val, noise_val)), batch_size=batch_size)

    # Delete intermediate tensors to free GPU/CPU memory
    del U_train, U_val, noise_train, noise_val

    return dataT, dataV

# function to train NN given data
def first_train_leaky(funct, dataT, dataV, nbBatchVal = 16, nbBatch = 128, totpochs = 50, hn = 4, lr = 1e-2, d = d, save_as = "plot"):
    """
    Trains a neural network model (LeakyNN_weight) on given data using Adam optimizer.

    Parameters:
        funct (callable): A function that computes the deterministic part of the target (used to generate Y = f(U) + N).
        dataT (DataLoader): Training dataset (U, noise).
        dataV (DataLoader): Validation dataset (U, noise).
        nbBatchVal (int): Number of validation batches.
        nbBatch (int): Number of training batches.
        totpochs (int): Total number of training epochs.
        hn (int): Hidden layer size of the neural network.
        lr (float): Learning rate for the optimizer.
        d (int): Input dimensionality.

    Returns:
        Q_best (nn.Module): The best model (lowest validation loss).
        lossHi (float): The best validation loss achieved.
        best (int): Epoch number corresponding to the best model.
    """
    #model_Q = LeakyNN_weight(d, hn, hn, 1).cuda(device) # load the NN model
    model_Q = LeakyNN(L).cuda(device)
    
    optimizer_Q = optim.Adam(model_Q.parameters(), lr)

    loss_plot = []
    loss_plot_t = []


    t0 = time.time()
    torch.cuda.synchronize()

    # ----- INITIAL VALIDATION LOSS (before training) -----
    lossHi = 0.
    with torch.no_grad() :
        for U, N in dataV:
            
            Yq = model_Q(U)
            dt = funct(U, d)
            dt = dt.view(-1, 1)
            Y = dt + N
            loss = loss_func(Y, Yq)
            lossHi += loss
        lossHi /=nbBatchVal
    print(f'Epoch [{0}/{totpochs}], Loss: {lossHi.item():.4f}')

    # -----MAIN TRAINING STEP-----
    for epoch in range(totpochs):

        # TRAINING STEP
        losstmp = 0.
        for U, N in dataT:
            optimizer_Q.zero_grad()
            Yq = model_Q(U)
            dt = funct(U, d)
            dt = dt.view(-1, 1)
            Y = dt + N
            loss = loss_func(Y, Yq)
            loss.backward()
            optimizer_Q.step()
            losstmp += loss
        losstmp /=nbBatch
        loss_plot_t.append(losstmp.item())

        # VALIDATION STEP
        losstmp = 0.
        with torch.no_grad() :
            for U, N in dataV:
                Yq = model_Q(U)
                dt = funct(U, d)
                dt = dt.view(-1, 1)
                Y = dt + N
                loss = loss_func(Y, Yq)
                losstmp += loss
            losstmp /=nbBatchVal
            print(f'Epoch [{epoch + 1}/{totpochs}], Loss: {losstmp.item():.4f}')
            loss_plot.append(losstmp.item())
            if losstmp < lossHi :
                best = epoch
                lossHi = losstmp
                Q_best = copy.deepcopy(model_Q)

    plt.figure()
    plt.plot(loss_plot, color='r', label='Validation loss estimation')
    plt.plot(loss_plot_t, color='b', label='Training loss estimation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()
    plt.savefig(save_as+".pdf")
    # plt.show()

    return Q_best, lossHi, best


# In[10]:


def conversion_weights(model):
    """
    Converts a PyTorch model's weights and biases into a flattened list or array,
    and prints them in a formatted way.

    Parameters:
        model (nn.Module): The trained neural network model.

    Returns:
        np.array: A 1D NumPy array containing all weights and biases in flattened order.
    """
    
    L = [] # List to store flattened weights and biases
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}:")
            print(param.data)
            print()
            
    # Count total parameters
    total_params = sum(p.numel() for p in model.parameters())

    final_list = []

    for layer in model.modules():
        # Get layer weights and biases as Python lists (for easy manipulation)
        if isinstance(layer, nn.Linear):
            weight = layer.weight.detach().cpu().numpy().tolist()
            bias   = layer.bias.detach().cpu().numpy().tolist()

            for w_row, b in zip(weight, bias):
                final_list.extend(w_row + [b])  # append weight row + corresponding bias
    
    for i in range(total_params):
        print('WB['+str(i)+'] = '+str(round(final_list[i],4))+'f;')
        L.append(round(final_list[i],4))
        
    return np.array(L)


# In[11]:


def save_checkpoint(model, path="checkpoint.pth"):
    """
    Saves the state (weights) of a PyTorch model to a file.

    Parameters:
        model (nn.Module): The trained model to be saved.
        path (str): File path where the checkpoint will be stored (default: "checkpoint.pth").
    """
    checkpoint = {
        'model_state_dict': model.state_dict()
                    }
    # Save the checkpoint to the specified file path
    torch.save(checkpoint, path)

def load_checkpoint(model, path="checkpoint.pth"):
    """
    Loads model weights from a previously saved checkpoint file.

    Parameters:
        model (nn.Module): The model instance to load weights into.
        path (str): Path to the checkpoint file.

    Returns:
        model (nn.Module): The model with loaded weights.
    """

    # Load the checkpoint from disk
    checkpoint = torch.load(path, weights_only=True, map_location=device)

    # Load the stored state dictionary into the model
    model.load_state_dict(checkpoint['model_state_dict'])

    return model


# In[12]:


def save_NN_path(funct, dim = d, ):
    """
    Trains a neural network on generated training/validation data for a given dimension,
    then saves the trained model parameters to a checkpoint file.

    Parameters:
        funct (callable): Function that defines the deterministic mapping f(U).
        dim (int): Dimensionality of the input data (default: uses global 'dim_try').

    Returns:
        None
    """
    
    dataT, dataV = data_tr_val(d = dim) # training and validation data
    Q_best_my_log, lossHi, _ = first_train_leaky(funct, dataT, dataV, d = dim, save_as=f"vis_{funct.__name__}_{d}")
    
    save_checkpoint(Q_best_my_log) # Saves the best-performing model’s weights to "checkpoint.pth"


# In[13]:


def evaluate_model(model, po, dimension = d, device = device):
    """
    Evaluates a neural network model on uniform distributed data of varying sizes.
    
    Parameters:
        model (torch.nn.Module): The neural network model on the device.
        po (int): Exponent used to generate sizes as 10**po.
        dimension (int): The dimensionality of the input data.
        device (torch.device): The device where the model is located (CPU/GPU).
    
    Returns:
        None: Plots the maximum scalar output as a function of size.
    """
    max_outputs = []
    sizes = [10**i for i in range(1, po+1)]
    
    # Generate the uniform random variables once for the largest size
    max_size = 10**po
    uniform_data = torch.rand(max_size, dimension, device=device)  # Uniform distribution [0, 1)
    
    for size in sizes:
        # Slice the uniform_data to get the first 'size' samples
        inputs = uniform_data[:size, :]  # Slice to get a batch of the desired size
        
        # Evaluate the model on the generated inputs
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            outputs = model(inputs)
        
        # Assuming the model output is scalar (single value per input)
        max_output = torch.max(outputs).item()  # Extract the maximum scalar value
        max_outputs.append(max_output)

    # Print the values of max_outputs
    print(f"Max scalar outputs for each size (10^i): {max_outputs}")
    
    # Plotting the results
    plt.figure()
    plt.plot(sizes, max_outputs, marker='o')
    plt.xscale('log')  # Set x-axis to log scale to represent sizes (10^i)
    plt.xlabel('Size (10^i)')
    plt.ylabel('Maximum Scalar Output')
    plt.title('Maximum Scalar Output of Model vs. Size')
    plt.grid(True)
    plt.savefig("evaluated_model.pdf")
    # plt.show()
    
def evaluate_model_batch(model, po, dimension = d, device = device, batch_size=128):
    """
    Evaluates a neural network model on uniform distributed data of varying sizes.
    
    Parameters:
        model (torch.nn.Module): The neural network model on the device.
        po (int): Exponent used to generate sizes as 10**po.
        dimension (int): The dimensionality of the input data.
        device (torch.device): The device where the model is located (CPU/GPU).
    
    Returns:
        None: Plots the maximum scalar output as a function of size.
    """
    max_outputs = []
    sizes = [10**i for i in range(1, po+1)]
    
    # Generate the uniform random variables once for the largest size
    max_size = 10**po
    uniform_data = torch.rand(max_size, dimension, device=device)  # Uniform distribution [0, 1)
    
    
    # for size in sizes:
    #     # Slice the uniform_data to get the first 'size' samples
    #     inputs = uniform_data[:size, :]  # Slice to get a batch of the desired size
        
    #     # Evaluate the model on the generated inputs
    #     model.eval()  # Set model to evaluation mode
    #     with torch.no_grad():
    #         outputs = model(inputs)
    
    #     # Assuming the model output is scalar (single value per input)
    #     max_output = torch.max(outputs).item()  # Extract the maximum scalar value
    #     max_outputs.append(max_output)
    
    model.eval()
    with torch.no_grad():
        for size in sizes:
            local_max = -float("inf")

            # Iterate over uniform_data[:size] in small batches
            for i in range(0, size, batch_size):
                batch = uniform_data[i:i+batch_size]

                # Forward pass
                out = model(batch)

                # Update running max
                batch_max = out.max().item()
                if batch_max > local_max:
                    local_max = batch_max

            max_outputs.append(local_max)

    # Print the values of max_outputs
    print(f"Max scalar outputs for each size (10^i): {max_outputs}")
    
    # Plotting the results
    plt.figure()
    plt.plot(sizes, max_outputs, marker='o')
    plt.xscale('log')  # Set x-axis to log scale to represent sizes (10^i)
    plt.xlabel('Size (10^i)')
    plt.ylabel('Maximum Scalar Output')
    plt.title('Maximum Scalar Output of Model vs. Size')
    plt.grid(True)
    plt.savefig("model_size_vs_output.pdf")
    # plt.show()
    
    del uniform_data
    




# ## 1. For $F(z) = \frac{1}{d} \sum_{i=1}^{d}\frac{log(z_i)}{log(z_i)-1}$
# 
# ## 2. For $F(z) = \frac{1}{d} \sum_{i=1}^{d} |sin (3 \pi  z_i) | $

# In[14]:


#----------------------------------------------------------
# Train, save, reload, and extract weights from a neural network
# ----------------------------------------------------------
# save_NN_path(my_log_dim, dim = d)
save_NN_path(n_g_trig_dim, dim = d)
# save_NN_path(loc_max_dim, dim = d)
# save_NN_path(n_g_dim, dim = d)

# load the model to compute cdf

#model_nn = LeakyNN_weight(d, hn, hn, 1).cuda(device)
model_nn = LeakyNN(L).cuda(device)
model_load = load_checkpoint(model_nn)

print("dimension = " + str(d))
weights = conversion_weights(model_load)


# In[15]:


evaluate_model(model_nn, 7, d, device)


# In[16]:


np.savetxt("weights.txt", weights)


# In[ ]:


import subprocess

if d == 2:
    subprocess.call("./create_exec2D.sh", shell=True)
    
elif d == 3:
    subprocess.call("./create_exec3D.sh", shell=True)
else:
    print("Invalid dimension. Please set dimension to 2 or 3.")


# In[22]:


subprocess.call("./run_exec.sh", shell=True)


# In[19]:


# TO profile 
# nsys profile -t nvtx,cuda --stats=true --force-overwrite true --wait=all -o my_report ./MC

