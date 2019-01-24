"""
Logistic Regression module to train logistic regression module.Happy Coding!!
"""
__author__ = "Mathew Sam"
__version__ = "0.0.4"
__email__ = "m1sam@eng.ucsd.edu"
__maintainer__ = "Mathew Sam"

import random 

import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import color,exposure
import pickle

from DataLoader import DataLoader

class LogisticRegression:
    def __init__(self,window_size,num_classes=1):
        """
        A window of pixel values are extracted and are used to calculate probability that the center pixel belongs to the class in question
        Args:
        ----
            window_size:window size of array provided for training at every step
        Kwargs:
        ------
            num_classes: number of classes in 
        """
        assert isinstance(window_size,int),"Window size must be an integer"
        assert isinstance(num_classes,int),"Number of classes must be an integer"

        self.num_classes = num_classes
        self.W = np.random.rand(window_size**2,num_classes)
        self.b = np.random.rand(1,num_classes)
        self.window_size = window_size


    @staticmethod
    def sigmoid(x,W,b):
        """
        Calculates the sigmoid given parameters and inputs
        Args:
        ----
            x:input data
            W:weight matrix
            b:bias vector(scalar value since 1/0 classificiation)
        Returns:
        -------
            y:probability of x belonging to a class
        """
        z = np.dot(x,W) + b
        z = np.exp(z - np.max(z))
        z = z/np.sum(z,1)
        return  z

    def train(self,sample_generator,epochs=5,learning_rate=0.001,epsilon = 0.00000001):
        """
        Args:
        ----
            self:pointer to current instance of the class
            sample_generator: generates samples for training
        Kwargs:
        ------
            epochs:number of epochs for training
            default value:5
            data type:int

            learning_rate:learning rate of model
            default value:0.001
            datatype: float
        """
        assert isinstance(epochs,int),"Number of epochs is an integer"
        assert isinstance(learning_rate,float),"Learning rate is a decimal"
        assert learning_rate<1
        sample_num = 0
        error_plot = [0]
        for epoch in range(epochs):
            for sample,label in sample_generator:
                y = self.sigmoid(sample.reshape(1,-1),self.W,self.b)
                gradient = label - y
                self.W = self.W + learning_rate*gradient*sample.reshape(-1,1)
                self.b = self.b + learning_rate*gradient
                sample_num = sample_num + 1
                if sample_num%1000==0:
                    CE = -label*np.log(y+epsilon) - (1-label)*np.log(1-y+epsilon)
                    error_plot.append(0.1*CE[0,0] + 0.9*error_plot[-1])
            if(epoch%10==0):
                print("Completed {} percent ".format(epoch*100//epochs)) 
        
        plt.plot(error_plot,label="Cross entropy")
        plt.xlabel("sample_num x1000")
        plt.ylabel("Cross entropy loss")
        plt.show()

    def test_image(self,input_image,color_space="YUV"):
        """
        Testing images using logistoic regression
        Args:
        ----
            self:pointer to current instance of the class
            input_image:input test image
        Kwargs:
        ------
            color_space: sets color space for use
            default_value:YUV
            data type:str

            confidence:set confidence of classification
            default_value:0.9
            data_type:float
        """
        mask = np.zeros((input_image.shape[0],input_image.shape[1],self.num_classes))
        image = color.convert_colorspace(input_image,"RGB",color_space)[:,:,1]
        image = (image - np.min(image))/(np.max(image) - np.min(image))
        image = exposure.adjust_gamma(image,gamma=0.5)
        for i in range(self.window_size//2,mask.shape[0]-(self.window_size//2)):
            for j in range(self.window_size//2,mask.shape[1]-(self.window_size//2)):
                window = image[i-(self.window_size//2):i+(self.window_size//2),j-(self.window_size//2):j+(self.window_size//2)]
                mask[i,j,:] = self.sigmoid(window.reshape(1,-1),self.W,self.b)
        return mask



if __name__ == "__main__":
    DM = DataLoader("ECE276A_HW1/trainset/",0.9,["barrel_blue","non_barrel_blue","rest"])
    window_size = 10
    gen = DM.data_generator("labeled_data/Stored_Values.pickle",window_size=window_size,step_size=2)
    model = LogisticRegression(window_size,num_classes=3)
    model.train(gen,epochs=1000,learning_rate=0.01)
    pickle_file = "trained_models/model2.pickle"
