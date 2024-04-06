import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as  transforms
from torchvision.datasets import CIFAR100, CIFAR10
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms.functional import resize
from torchvision.transforms import CenterCrop
import torch.optim as optim
from torchvision.transforms import ToTensor
from torchvision.io import read_image
from mlp import *
# from mlp_d import *

# Seed the random number generator for all devices (both CPU and CUDA)
torch.manual_seed(0)

# evaluates the trained model
def evaluate(model, loader):

    # we need to switch the model into the evaluation mode
    model.eval()

    # create a list to store the prediction results
    res_store = []
    for batch in loader:
        x, y = batch

        # make a prediction for a data sample "x"
        pred = model(x)
        pred = (pred > 0.5).float().squeeze(1)
        y = y.squeeze(1)

        # if the prediction is correct, append True; else append False
        res_store += (pred == y).tolist()

    # return the classification accuracy
    acc = sum(res_store)/len(res_store)
    return acc

# Specify Network-related hyper-parameters 
# data_path = "./simpleDataGen/highData.txt"             # specify location of data

with open('./simpleDataGen/highData.txt', 'r') as file:
    high = file.read()
# Split the content into a list of elements separated by comma
highFid = high.split(',')

# Convert elements to integers (or floats if needed)
data = [float(highFid_i) for highFid_i in highFid]
data = torch.Tensor(data)

input_dim = 4                           # equal to number of
hidden_dim = 20                         # number of hidden neurons
output_dim = 4                          # number of output neurons
layers = [64,64]
device = 'cpu'                          # use CPU 
activations = 'Tanh'
# activations = 'ReLU'
init_type = 'Xavier normal'
# batch_size = 256                        # specify batch size

mlp_H_nl = MLP(input_dim, output_dim, layers, activations, init_type)

# Split the data into training set, validation set and test set
train_set_size = 0.6
val_set_size = 0.2
test_set_size = 0.2
train_set, val_set, test_set = torch.utils.data.random_split(data,[train_set_size,val_set_size,test_set_size])

# Wrap the dataset into Pytorch dataloader to pass samples in "minibatches"
# train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=False)
# val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=True, drop_last=False)
# test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=False)

# Specify Training-Related hyper params
save = "best_model"
epochs = 50             # specify number of epochs
learning_rate = 0.01    # specify learning rate

# loss function
## MSE for NN_L
def MSE_yL(y, y_star):
    return torch.mean(torch.square(torch.abs(torch.sub(y,y_star))))

## MSE for NN_H
def MSE_yH(y, y_star):
    return torch.mean(torch.square(torch.abs(torch.sub(y,y_star))))

acc_best = 0.0
training_loss = []

# Run training 
for epoch in range(epochs):
    mlp_H_nl.train()
    print(f"epoch:{epoch}")

    # Iterate batches in dataloader
    # for batch in train_dataloader:
        # x, y_star = batch

        # y = mlp(x)
        # cost = MSE(y,y_star) # Calculate cost
        # d_cost_d_y = MSE_prime(y,y_star)

        # training_loss.append(cost)

        # # Perform a single optimization step (weights update)
        # mlp.backward(x,d_cost_d_y,y,learning_rate)
    
    x_H = [0,0.4,0.6,1]
    x_H = torch.Tensor(x_H)
    # print(x_H.size())
    x_H = torch.t(x_H)
    # print(x_H.size())
    y_star = torch.t(data)
    # print(y_star.size())

    y = mlp_H_nl(x_H)
    # print(y.size())
    cost = MSE_yH(y,y_star) # Calculate cost
    d_cost_d_y = MSE_prime(y,y_star)

    # total cost <- cost 1 + cost 2 

    training_loss.append(cost)

    # Perform a single optimization step (weights update)
    # Define a loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(mlp_H_nl.parameters(), lr=0.01, momentum=0.9)
    # optimizer = optim.Adam(mlp_H_nl.parameters(), lr=0.001)

    # Forward pass
    #loss = criterion(y, y_star)
    loss = MSE_yH(y, y_star)

    # Backward pass and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch+1) % 5 == 0:
        print("Loss:", loss.item())

    # Evaluate the model
    # acc = evaluate(mlp, val_dataloader)
    # acc = evaluate(mlp_H_nl, data)

    # # Save the current weights if the accuracy is better in this iteration
    # if acc > acc_best and save:
    #     torch.save(mlp_H_nl.W1, save + "_W1")
    #     torch.save(mlp_H_nl.W2, save + "_W2")

    # if (epoch+1) % 5 == 0: <- use this if you want to print the validation accuracy every 5 epochs
    # print(f"Epoch: #{epoch+1}: validation accuracy = {acc*100:.2f}%; training loss = {torch.mean(torch.tensor(training_loss))}")