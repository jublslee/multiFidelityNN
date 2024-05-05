import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as  transforms
from torchvision.datasets import CIFAR100, CIFAR10
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms.functional import resize
from torchvision.transforms import CenterCrop
from torchvision.transforms import ToTensor
from torchvision.io import read_image
# from torchsummary import summary
from tqdm import tqdm
import numpy as np
import pandas as pd

class CNN(nn.Module):

    def __init__(self, numChannels, numClasses):
        super(CNN, self).__init__()
        self.classes = numClasses

        #######################################################
        # *** TASK 1 ***
        # Define your "Lego bricks": all the layers of your CNN
        # you will be using later in the "forward" function
        # pytorch implementation: nn.Conv2d
        # you can check https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html

        # If you haven't done so before, consider playing with this demo of CNNs:
        # https://poloclub.github.io/cnn-explainer/

        #######################################################
        # *** THINK *** What is the relationship between output feature channel and number of kernels?
        # *** THINK *** Can you draw a picture to describe the relationship among input size,
        #               kernel size and output size in a convolution layer?
        #######################################################
#torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
        # Convolutional layers:
        self.conv1 = nn.Conv2d(in_channels = numChannels, out_channels = 96, kernel_size=(11, 11), stride=(4, 4))
        self.conv2 = nn.Conv2d(in_channels = 96, out_channels = 256, kernel_size=(5,5), stride=(1, 1))
        self.conv3 = nn.Conv2d(in_channels = 256, out_channels = 384, kernel_size=(3,3), stride=(1, 1))
        self.conv4 = nn.Conv2d(in_channels = 384, out_channels = 384, kernel_size=(3,3), stride=(1, 1))
        self.conv5 = nn.Conv2d(in_channels = 384, out_channels = 256, kernel_size=(3,3), stride=(1, 1))

        # Activation function:
        # check https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
        self.relu = nn.ReLU()

        # Pooling layer:
        # check https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # Batch normalization layers:
        self.batchnorm1 = nn.BatchNorm2d(num_features=96)
        self.batchnorm2 = nn.BatchNorm2d(num_features=256)

        # Fully-connected layers:
        self.fc1 = nn.Linear(in_features = 2304, out_features=1024)
        self.fc2 = nn.Linear(in_features = 1024, out_features=10)

        #######################################################
        # *** THINK *** We have very specific numbers of neurons (2304 and 1024)
        #               in fully-connected layers (fc1 and fc2). Make sure you understand
        #               where these numbers are comming from (and when they can be different).
        #######################################################

        #######################################################
        # *** THINK *** As we are using CrossEntropy loss during training, so a softmax activation
        #               function (= implemented as a softmax layer to apply it to the whole layer)
        #               is not needed here since it's already included in Pytorch's implementation
        #               of cross-entropy loss. See this for details:
        #               https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        #######################################################

    # Evaluation function
    def evaluate(self, model, dataloader, classes, device):

        # We need to switch the model into the evaluation mode
        model.eval()

        # Prepare to count predictions for each class
        correct_pred = {classname: 0 for classname in classes}
        total_pred = {classname: 0 for classname in classes}

        # For all test data samples:
        for data in dataloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)

            images = images.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
            predictions = predictions.detach().cpu().numpy()

            # Count the correct predictions for each class
            for label, prediction in zip(labels, predictions):

                # If you want to see real and predicted labels for all samples:
                # print("Real class: " + classes[label] + ", predicted = " + classes[prediction])

                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

        for c in classes:
            print(c,": accuracy is ", correct_pred[c]/total_pred[c])

        # Calculate the overall accuracy on the test set
        acc = sum(correct_pred.values()) / sum(total_pred.values())

        return acc


    def forward(self, x):

        x = resize(x, size=[256])


        #######################################################
        # *** TASK 1 *** Now use your "Lego bricks" to build the CNN network
        #                by defining all operations in a correct order:

        # Convolutional, ReLU, MacPooling and Batchnorm layers go first
        # (see the slides with instructions for the network architecture)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        self.batchnorm1

        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        self.batchnorm2

        x = self.conv3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.relu(x)

        x = self.conv5(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # *** THINK *** What if we remove one of the layers? Will the network still work?

        # After the last pooling operation, and before the first
        # fully-connected layer, we need to "flatten" our tensors
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)

        # *** THINK *** Can fully-connected layers accept data if they are not flattened?

        # Finally, we need our two-layer perceptron (two fully-connected layers) at the end of the network:
        #x = torch.flatten(x, 1)
        x = self.fc2(x)
        #x = self.relu(x)


        return x

        #
        #######################################################