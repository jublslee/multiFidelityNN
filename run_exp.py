######### Import Packages #########
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler, Subset
import torch.optim as optim
from torch.utils.data import TensorDataset
from deepNN import *
from MLP import *
from exCases import *

# Seed the random number generator for all devices (both CPU and CUDA)
torch.manual_seed(0)

#################################################################################

######### Define functions #########
#### dataConvert: converts data to vector from txt file ####
## @param dataPath
##        path to data
# ## @return data converted to vector 
# def dataConvert(dataPath):
#     with open(dataPath, 'r') as file:
#         dat = file.read()
#     # Split the content into a list of elements separated by comma
#     datVec = dat.split(',')
#     return datVec

### MSE: loss function ####
## @param y 
##        output value of MLP network
## @param y_star
##        actual y value
## @return output of MSE function
def MSE(y, y_star):
    return torch.mean(torch.square(torch.abs(torch.sub(y,y_star))))

# evaluates the trained model
def evaluate(model, load_L, load_H):
    # we need to switch the model into the evaluation mode
    model.eval()

    # create a list to store the prediction results
    res_store = []
    for batchL, batchH in zip(load_L, load_H):
        xL, yL = batchL
        xH, yH = batchH

        predL = model.lf_models[0](xL)
        predL = (predL > 0.5).float().squeeze(1)
        yL = yL.squeeze(1)

        predH = model.hf_prediction(xH)
        predH = (predH > 0.5).float().squeeze(1)
        yH = yH.squeeze(1)

        # if the prediction is correct, append True; else append False
        res_store += ((abs(predH-yH) + abs(predL-yL)) * 0.5 <= 5).tolist()

    # return the accuracy
    acc = sum(res_store)/len(res_store)
    return acc

# #################################################################################

### Training
input_dim = 11                           # equal to number of
output_dim = 11                          # number of output neurons
layers = [64,64]
device = 'cpu'                          # use CPU 
activations = 'Tanh'
# activations = 'ReLU'
init_type = 'Xavier normal'
# batch_size = 256                        # specify batch size

# mlp_L = MLP(input_dim, output_dim, layers, activations, init_type)
lf_list = [LF_MK1()]

num_lf_models = len(lf_list)
lf_nn_sizes = [{ 'num_hidden_layers': 4, 'num_neurons': 20, }
                for ii in range(num_lf_models)]
nl_nn_size = {
    'num_hidden_layers': 4,
    'num_neurons': 20,
}

mfnn = MLP(lf_list, HF_MK1(), lf_nn_sizes, nl_nn_size)

# Generate data
data = torch.linspace(0.0, 1.0, 100).unsqueeze(1)  

# Split the data into training set, validation set and test set
train_set_size = 0.6
val_set_size = 0.2
test_set_size = 0.2

# Define the sizes of each set (you can adjust these as needed)
train_size = int(train_set_size * len(data))
val_size = int(val_set_size * len(data))
test_size = int(test_set_size * len(data))

# Assuming you also have corresponding target labels, otherwise, generate them as well
# For simplicity, let's create some dummy target labels
targets_L = LF_MK1().eval(data)
targets_H = HF_MK1().eval(data)

# Split the data indices into training, validation, and test sets
indices = list(range(len(data)))
np.random.shuffle(indices)
train_indices, val_indices, test_indices = indices[:train_size], indices[train_size:train_size+val_size], indices[train_size+val_size:]

# Create SubsetRandomSampler for each set
train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)
test_sampler = SubsetRandomSampler(test_indices)

# Create DataLoader objects for each set
batch_size = 20  # Adjust as needed
train_loader_L = DataLoader(TensorDataset(data, targets_L), batch_size=batch_size, sampler=train_sampler)
val_loader_L = DataLoader(TensorDataset(data, targets_L), batch_size=batch_size, sampler=val_sampler)
test_loader_L = DataLoader(TensorDataset(data, targets_L), batch_size=batch_size, sampler=test_sampler)
train_loader_H = DataLoader(TensorDataset(data, targets_H), batch_size=batch_size, sampler=train_sampler)
val_loader_H = DataLoader(TensorDataset(data, targets_H), batch_size=batch_size, sampler=val_sampler)
test_loader_H = DataLoader(TensorDataset(data, targets_H), batch_size=batch_size, sampler=test_sampler)

# Specify Training-Related hyper params
save = "best_model"
epochs = 100             # specify number of epochs
learning_rate = 0.001    # specify learning rate

acc_best = 0.0
training_loss = []

for epoch in range(epochs):
    mfnn.train()
    print(f"epoch:{epoch}")

    # Iterate batches in dataloader
    # for batch in train_loader:
    beta = 0
    for batchL, batchH in zip(train_loader_L, train_loader_H):
        x_L, y_star_L = batchL
        y_L = mfnn.lf_models[0](x_L)

        x_H, y_star_H = batchH
        y_H = mfnn.hf_prediction(x_H)

        # beta = beta + mfnn.hf_l.weight
        # print(mfnn.hf_nl.weights)
        beta = [param for param in mfnn.parameters() if param.requires_grad]

        # Lambda is the regularization strength
        lambda_reg = 0.01

        # Regularization term
        regularization_term = lambda_reg * sum(torch.sum(param ** 2) for param in beta)

        criterion = torch.nn.CrossEntropyLoss()
        # optimizer = optim.SGD(mfnn.parameters(), lr=learning_rate, momentum=0.9)
        optimizer = optim.Adam(mfnn.parameters(), lr=learning_rate)
        loss_L = MSE(y_L,y_star_L) # Calculate cost
        loss_H = MSE(y_H,y_star_H) # Calculate cost
        # total_loss = (loss_L + loss_H)
        total_loss = (loss_L + loss_H + regularization_term)

        # training_loss.append(loss_L + loss_H)
        training_loss.append(total_loss)

        # Perform a single optimization step (weights update)
        # mlp.backward(x,d_cost_d_y,y,learning_rate)
        optimizer.zero_grad()
        total_loss.backward()
        # loss_L.backward()
        # loss_H.backward()
        optimizer.step()
        if (epoch+1) % 10 == 0:
            print("Loss:", total_loss.item())

    # Evaluate the model
    acc = evaluate(mfnn, val_loader_L, val_loader_H)
    # Save the current weights if the accuracy is better in this iteration
    if acc > acc_best and save:
        torch.save(mfnn.hf_l.weight, save + "_hf_1_W")
        torch.save(mfnn.hf_nl.weights, save + "_hf_n1_W")
        # torch.save(mfnn.lf_models[1].weights, save + "_lf_W")
        for lf_i in range(len(lf_list)):
            if isinstance(mfnn.lf_models[lf_i], nn.Linear):
                torch.save(mfnn.lf_models[lf_i].weight, save + "_lf_" + str(lf_i) + "_W")
            else:
                torch.save(mfnn.lf_models[lf_i].weights, save + "_lf_" + str(lf_i) + "_W")
        acc_best = acc
    print(f"Epoch: #{epoch+1}: validation accuracy = {acc*100:.2f}%; training loss = {torch.mean(torch.tensor(training_loss))}")

# load the model’s weights
mfnn.hf_l.weight = torch.load(save + "_hf_1_W")
mfnn.hf_nl.weights = torch.load(save + "_hf_n1_W")
for lf_i in range(len(lf_list)):
    mfnn.lf_models[lf_i].weight = torch.load(save + "_lf_" + str(lf_i) + "_W")
acc = evaluate(mfnn, test_loader_L, test_loader_H)
print(f"Test accuracy = {acc}")

x  = torch.linspace(0.0,1.0,100)
data = torch.linspace(0.0, 1.0, 100).unsqueeze(1)
hf = HF_MK1().eval(x)
lf = LF_MK1().eval(x)

plt.plot(x.numpy(),hf.numpy(),label='High-Fidelity')
plt.plot(x.numpy(),lf.numpy(),label='Low-Fidelity')
# predVals = mfnn.lf_models[0](data)
predVals = mfnn.hf_prediction(data)
plt.plot(x.numpy(),predVals.detach().numpy(),label='Multi-Fidelity Prediction')
plt.legend()
plt.show()