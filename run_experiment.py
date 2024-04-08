######### Import Packages #########
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
from func import *
from torch.utils.data import TensorDataset

# Seed the random number generator for all devices (both CPU and CUDA)
torch.manual_seed(0)

#################################################################################

######### Define functions #########
#### dataConvert: converts data to vector from txt file ####
## @param dataPath
##        path to data
## @return data converted to vector 
def dataConvert(dataPath):
    with open(dataPath, 'r') as file:
        dat = file.read()
    # Split the content into a list of elements separated by comma
    datVec = dat.split(',')
    return datVec

### MSE: loss function ####
## @param y 
##        output value of MLP network
## @param y_star
##        actual y value
## @return output of MSE function
def MSE(y, y_star):
    return torch.mean(torch.square(torch.abs(torch.sub(y,y_star))))

#################################################################################

highFid = dataConvert('./simpleDataGen/highData.txt')
lowFid = dataConvert('./simpleDataGen/lowData.txt')

# Convert elements to integers (or floats if needed)
data_H = [float(highFid_i) for highFid_i in highFid]
data_H = torch.Tensor(data_H)

data_L = [float(lowFid_i) for lowFid_i in lowFid]
data_L = torch.Tensor(data_L) 


### Training
input_dim = 11                           # equal to number of
output_dim = 11                          # number of output neurons
layers = [64,64]
device = 'cpu'                          # use CPU 
activations = 'Tanh'
# activations = 'ReLU'
init_type = 'Xavier normal'
# batch_size = 256                        # specify batch size

mlp_L = MLP(input_dim, output_dim, layers, activations, init_type)

# Split the data into training set, validation set and test set
train_set_size = 0.6
val_set_size = 0.2
test_set_size = 0.2
train_set, val_set, test_set = torch.utils.data.random_split(data_H,[train_set_size,val_set_size,test_set_size])

# Wrap the dataset into Pytorch dataloader to pass samples in "minibatches"
# train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=False)
# val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=True, drop_last=False)
# test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=False)

# Specify Training-Related hyper params
save = "best_model"
epochs = 50             # specify number of epochs
learning_rate = 0.01    # specify learning rate

y_L = 0

# Run training 
for epoch in range(epochs):
    mlp_L.train()
    print(f"epoch:{epoch}")

    x_L = [i / 10 for i in range(11)]
    x_L = torch.Tensor(x_L)
    y_star = torch.t(data_L)

    y = mlp_L(x_L)

    # Perform a single optimization step (weights update)
    # Define a loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(mlp_L.parameters(), lr=0.01, momentum=0.9)
    # optimizer = optim.Adam(mlp_H_nl.parameters(), lr=0.001)

    # Forward pass
    loss_L = MSE(y,y_star) # loss calculation for low fidelity network
    # loss_L = MSE_yH(y,y_star) # loss calculation for low fidelity network
    # training_loss.append(loss_L)

    # Backward pass and optimize
    optimizer.zero_grad()
    #loss_L.backward(retain_graph=True)
    loss_L.backward()
    optimizer.step()
    if (epoch+1) % 5 == 0:
        print("Loss:", loss_L.item())

    y_L = y

print(y_L)

###### NN_H

epochs = 50             # specify number of epochs
input_dim = 15                           # equal to number of
hidden_dim = 20                         # number of hidden neurons
output_dim = 4                          # number of output neurons
layers = [64,64]
device = 'cpu'                          # use CPU 
activations = 'Tanh'
# activations = 'ReLU'
init_type = 'Xavier normal'
# batch_size = 256                        # specify batch size
mlp_H_nl = MLP(input_dim, output_dim, layers, activations, init_type)

x = [0,0.4,0.6,1]
x = torch.Tensor(x)
x = torch.t(x)
x_H = torch.cat((x, y_L), dim=0)
y_star = torch.t(data_H)

# Run training 
for epoch in range(epochs):
    mlp_H_nl.train()
    print(f"epoch:{epoch}")

    y = mlp_H_nl(x_H)
    # print(y.size())
    cost = MSE(y,y_star) # Calculate cost
    # d_cost_d_y = MSE_prime(y,y_star)

    # total cost <- cost 1 + cost 2 

    # training_loss.append(cost)

    # Perform a single optimization step (weights update)
    # Define a loss function and optimizer
    # criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(mlp_H_nl.parameters(), lr=0.01, momentum=0.9)
    # optimizer = optim.Adam(mlp_H_nl.parameters(), lr=0.001)

    # Forward pass
    #loss = criterion(y, y_star)
    loss = MSE(y, y_star)


## test accuracy
    
x_test = [0.1,0.7,0.3,0.5,0.9]
y_test = cont_linCorr_H(x_test)
test_data = torch.tensor([
    [x_test[0], y_test[0]],
    [x_test[1], y_test[1]],
    [x_test[2], y_test[2]],
    [x_test[3], y_test[3]],
    [x_test[4], y_test[4]],
])

# switch the model into the evaluation mode
mlp_H_nl.eval()
# res_store = []
# for row in test_data:
#     x, y = row[[0,1]]
    
#     # make a prediction for a data sample "x"
#     pred = mlp_H_nl(x)
#     # pred = (pred > 0.5).float().squeeze(1)
#     # y = y.squeeze(1)

#     res_store += MSE(pred,y)
# acc = sum(res_store)/len(res_store)
# print(acc)

# sumError = 0
# total = 0
# with torch.no_grad():  # Disable gradient tracking
#     for row in test_data:
#         x, y = row[[0,1]]
#         outputs = mlp_H_nl(x)
#         _, predicted = torch.max(outputs.data, 1)
#         total += y.size(0)
#         sumError += MSE(predicted, y)

# #accuracy = 100 * correct / total
# accuracy = sumError/total
# print(f'Accuracy on the test set: {accuracy:.2f}%')


# for epoch in range(epochs):
#     print(f"epoch:{epoch}")
#     mlp_L.train()

#     # Iterate batches in dataloader
#     x_L = [i / 10 for i in range(11)]
#     x_L = torch.Tensor(x_L)
#     y_star = torch.t(data_L)

#     y = mlp_L(x_L)

#     # Perform a single optimization step (weights update)
#     # Define a loss function and optimizer
#     criterion = torch.nn.CrossEntropyLoss()
#     optimizer = optim.SGD(mlp_L.parameters(), lr=0.01, momentum=0.9)
#     # optimizer = optim.Adam(mlp_H_nl.parameters(), lr=0.001)

#     # Forward pass
#     loss_L = MSE(y,y_star) # loss calculation for low fidelity network

#     # Backward pass and optimize
#     optimizer.zero_grad()
#     #loss_L.backward(retain_graph=True)
#     loss_L.backward()
#     optimizer.step()
#     x.detach()
#     y_star.detach()

#     mlp_H_nl.train()

#     # Iterate batches in dataloader
#     for batch in train_dataloader:

#         x, y_star = batch

#         y = mlp(x)

#         cost = huberLoss(y,y_star) # Calculate cost
#         d_cost_d_y = huber_prime(y,y_star)
#         training_loss.append(cost)

#         # Perform a single optimization step (weights update)
#         mlp.backward(x,d_cost_d_y,y,learning_rate)

#         x.detach()
#         y_star.detach()

#     # Evaluate the model
#     acc = evaluate(mlp_H, val_dataloader)

#     # Save the current weights if the accuracy is better in this iteration
#     if acc > acc_best and save:
#         torch.save(mlp.W1, save + "_W1")
#         torch.save(mlp.W2, save + "_W2")
#         torch.save(mlp.W3, save + "_W3")

#     # if (epoch+1) % 5 == 0: <- use this if you want to print the validation accuracy every 5 epochs
#     print(f"Epoch: #{epoch+1}: validation accuracy = {acc*100:.2f}%; training loss = {torch.mean(torch.tensor(training_loss))}")
    