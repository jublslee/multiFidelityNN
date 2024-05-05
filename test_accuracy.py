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

# Seed the random number generator for all devices (both CPU and CUDA)
torch.manual_seed(0)

#################################################################################

######### Define functions #########
# evaluates the trained model
def evaluate(model, x_test, y_test):

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

#################################################################################
epochs = 50             # specify number of epochs
learning_rate = 0.01    # specify learning rate
# batch_size = 256          
# acc_best = 0.0
# training_loss = []
# train_set_size = 0.6
# val_set_size = 0.2
# test_set_size = 0.2
# train_set, val_set, test_set = torch.utils.data.random_split(data,[train_set_size,val_set_size,test_set_size])

# # Wrap the dataset into Pytorch dataloader to pass samples in "minibatches"
# train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=False)
# val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=True, drop_last=False)
# test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=False)
mlp_L = MLP(input_dim=11, output_dim=11, layers=[64,64], activations='Tanh', init_type='Xavier normal')
mlp_H_nl = MLP(input_dim=15, output_dim=4, layers=[64,64], activations='Tanh', init_type='Xavier normal')

for epoch in range(epochs):
    print(f"epoch:{epoch}")
    mlp_L.train()

    # Iterate batches in dataloader
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

    # Backward pass and optimize
    optimizer.zero_grad()
    #loss_L.backward(retain_graph=True)
    loss_L.backward()
    optimizer.step()
    x.detach()
    y_star.detach()

    mlp_H_nl.train()

    # Iterate batches in dataloader
    for batch in train_dataloader:

        x, y_star = batch

        y = mlp(x)

        cost = huberLoss(y,y_star) # Calculate cost
        d_cost_d_y = huber_prime(y,y_star)
        training_loss.append(cost)

        # Perform a single optimization step (weights update)
        mlp.backward(x,d_cost_d_y,y,learning_rate)

        x.detach()
        y_star.detach()

    # Evaluate the model
    acc = evaluate(mlp, val_dataloader)

    # Save the current weights if the accuracy is better in this iteration
    if acc > acc_best and save:
        torch.save(mlp.W1, save + "_W1")
        torch.save(mlp.W2, save + "_W2")
        torch.save(mlp.W3, save + "_W3")

    # if (epoch+1) % 5 == 0: <- use this if you want to print the validation accuracy every 5 epochs
    print(f"Epoch: #{epoch+1}: validation accuracy = {acc*100:.2f}%; training loss = {torch.mean(torch.tensor(training_loss))}")
    