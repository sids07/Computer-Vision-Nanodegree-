## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 32, 5)
        #output = (224-5)/1 +1 = 220 --> (32,220,220)
        self.pool = nn.MaxPool2d(2,2)
        #(32,110,110)
        self.conv2 = nn.Conv2d(32,64,5)
        #(110-5)/1 +1 = 106 --> (64,106,106) after pool (64,53,53)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64*53*53,224)
        self.batch_norm1 = nn.BatchNorm1d(224)
        self.fc1_dropout = nn.Dropout(p=0.4)
        
        self.fc2 = nn.Linear(224,136)
        
        
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(self.batch_norm2(F.relu(self.conv2(x))))
        
        x = x.view(x.size(0),-1)
        #flatten
        x = self.fc1_dropout(self.batch_norm1(self.fc1(x)))
        x = self.fc2(x)
        
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
