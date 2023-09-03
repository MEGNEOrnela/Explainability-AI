
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from torchvision import models

from torch.utils import data

from torchvision import datasets
from PIL import Image





class CAM(nn.Module):

    '''
    This class implement the cam approach to vizualize the salient parts of an image for a class prediction.
    
    '''

    def __init__(self, model):
        super(CAM, self).__init__()
        
        self.resnet = model
        
        # Access the last convolutional layer of the model
        self.features_conv = nn.Sequential(*list(self.resnet.children())[:-2])
        
        # Get the classifier of the model
        self.classifier = self.resnet.fc
    
    def forward(self, x):
        x = self.features_conv(x)
        
        # Global Average Pooling (GPA) layer
        x = torch.mean(x, dim=[2, 3])
        
        x = self.classifier(x)
        return x




class GradCAM(nn.Module):
    '''
    This class implement the grad-cam approach to vizualize the salient parts of an image for a class prediction.
    Some methods:
    - register_hook() is call  on a tensor x (representing an image) and provide  with the 
    function activations_hook() that we want to execute when gradients are computed for x.
    - get_activations_gradient()  returns the stored gradients when needed.

    '''


    def __init__(self,model):
        super(GradCAM, self).__init__()
        
        self.resnet = model
        
        # Access the last convolutional layer of the model
        self.features_conv = nn.Sequential(*list(self.resnet.children())[:-2])
        
        # get the classifier of the model
        self.classifier = self.resnet.fc

        # Global Average Pooling (GAP) layer
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.gradients = None
    
    # custom function that will be executed when gradients are computed 
    def activations_hook(self, grad):
        self.gradients = grad
        
    def forward(self, x):
        x = self.features_conv(x)
        
        # register the hook
        h = x.register_hook(self.activations_hook)
        
        x = self.global_avg_pool(x)
        x = x.squeeze()
        x = self.classifier(x)
        return x
    
    # method to extract the gradient
    def get_activations_gradient(self):
        return self.gradients
    
    # method to get the last layer of the resnet
    def get_activations(self, x):
        return self.features_conv(x)
