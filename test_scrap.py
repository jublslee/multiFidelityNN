import torch

data = torch.tensor([
    [1, 2],
    [5, 6],
    [9, 10],
    [13, 1],
    [17, 18]
])

print(data)

# print(selected_columns)

from torch.utils.data import DataLoader, TensorDataset

dataset = TensorDataset(data) 
loader = DataLoader(dataset, batch_size=1, shuffle=True)

# for batch in loader:
    # Each batch is a tuple of (data,)
    # batch_data = batch[[0,1]]
    # print(batch_data)
    # Select specific columns from the batch
    # print(selected_batch_columns)
    # x, y = batch
    # print(batch)

for row in data:
    # Using only the selected columns (1 and 3 in this case)
    selected_features = row[[0,1]]
    print(selected_features)
    x, y = selected_features
    print(x)

from func import *
x_test = [0.1,0.7,0.3,0.5,0.9]
y_test = cont_linCorr_H(x_test)
test_data = torch.tensor([
    [x_test[0], y_test[0]],
    [x_test[1], y_test[1]],
    [x_test[2], y_test[2]],
    [x_test[3], y_test[3]],
    [x_test[4], y_test[4]],
])

for row in test_data:
    x, y = row[[0,1]]
    print(x)

print(test_data)