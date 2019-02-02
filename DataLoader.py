"""
This file serves to load data from the train set for labeling purposes. To make code generalizable, the DataLoader class can be 
imported and used  following the documentation provided. Happy coding!!! 
"""
__author__ = "Mathew Sam"
__version__ = "1.0.0"
__email__ = "m1sam@eng.ucsd.edu"
__maintainer__ = "Mathew Sam"

import os
from pathlib import Path
import pickle

import random

import numpy as np 
from skimage import color,exposure
from roipoly import RoiPoly
import cv2

import matplotlib.pyplot as plt

class DataLoader:
    def __init__(self,data_source,train_split,classes):
        """
        Args:
        ----
            self:pointer to current instance of the class
            datasource:folder containing files for training and validation
            train_split:fraction of the entire data that goes to training the model
            classes:classes alloted for classification
        """

        assert isinstance(data_source,str),"Please enter root directory of files"
        assert isinstance(train_split,float),"Please enter appropriate datatype"
        assert train_split<=1 and train_split>=0,"Please issue sensible split of data"
        assert isinstance(classes,list),"Classes need to be a list of class names"
        assert all(isinstance(x,str) for x in classes),"Name the classes for sanity's sake"

        Files = os.listdir(data_source)
        train_size = int(len(Files)*train_split)
        self.num_files = len(Files)
        self.root_location = data_source#Defines root location where files are located for ease of access
        self.train_files = Files[:train_size]#Defines set of files provided for training purposes
        self.validation_files = Files[train_size:]#Defines set of files provided for validation
        self.classes = classes
    
    def display_color_spaces(self,color_space="RGB"):
        """
        Displays color spaces for random image in the training set. Also prints statistics of the images in question.
        Args:
        ----
            self:pointer to current instance of the class
        Kwargs:
        ------ 
            color_space:color space to convert image to
            default:RGB
            datatype:str
            acceptable values:‘RGB’, ‘HSV’, ‘RGB CIE’, ‘XYZ’, ‘YUV’, ‘YIQ’, ‘YPbPr’, ‘YCbCr’, ‘YDbDr’
        """

        assert color_space in ['RGB', 'HSV', 'RGB CIE', 'XYZ', 'YUV', 'YIQ', 'YPbPr', 'YCbCr', 'YDbDr'],"Non acceptable colorspace"

        #image_file = random.choice(self.train_files)
        for image_file in self.train_files:
            image = plt.imread(self.root_location + image_file)

            print("Image dimensions:{}".format(image.shape))

            if(color_space!="RGB"):
                img = color.convert_colorspace(image,"RGB",color_space)
                img[:,:,1] = (img[:,:,1] - np.min(img[:,:,1]))/(np.max(img[:,:,1]) - np.min(img[:,:,1]))
                img[:,:,1] = exposure.adjust_gamma(img[:,:,1],gamma=0.4)
                img[:,:,2] = (img[:,:,2] - np.min(img[:,:,2]))/(np.max(img[:,:,2]) - np.min(img[:,:,2]))
                plt.subplot(2,2,1)
                plt.imshow(image),plt.xticks([]),plt.yticks([]),plt.title("Original Image")
                plt.subplot(2,2,2)
                plt.imshow(img[:,:,0],cmap='gray'),plt.xticks([]),plt.yticks([]),plt.title("Y")
                plt.subplot(2,2,3)
                plt.imshow(img[:,:,1],cmap='gray'),plt.xticks([]),plt.yticks([]),plt.title("U")
                plt.subplot(2,2,4)
                plt.imshow(img[:,:,2],cmap='gray'),plt.xticks([]),plt.yticks([]),plt.title("V")
                plt.show()
            else:
                img = image 
                plt.subplot(2,2,1)
                plt.imshow(img),plt.xticks([]),plt.yticks([]),plt.title("Original Image")
                plt.subplot(2,2,2)
                plt.imshow(img[:,:,0],cmap='gray'),plt.xticks([]),plt.yticks([]),plt.title("R")
                plt.subplot(2,2,3)
                plt.imshow(img[:,:,1],cmap='gray'),plt.xticks([]),plt.yticks([]),plt.title("G")
                plt.subplot(2,2,4)
                plt.imshow(img[:,:,2],cmap='gray'),plt.xticks([]),plt.yticks([]),plt.title("B")
                plt.show()
            plt.close()

    def draw_roi(self):
        """
        This function is meant to "test drive" the roipoly interface since it is new to the developer
        Args:
        ----
            self:pointer to current instance of the class
        """
        image_file = random.choice(self.train_files)
        image = plt.imread(self.root_location + image_file)
        plt.figure(1)
        plt.imshow(image)
        my_roi = RoiPoly(color='r')
        mask = my_roi.get_mask(image[:,:,2])*1
        plt.figure(2)
        plt.imshow(mask,cmap="gray")
        plt.show()
    
    def Labeling_module(self,pickle_file,color_space="YUV"):
        """
        Module to label data in consideration.
        Module builds a list of dictionaries with a key marking input image, color_space representation and label black and white image. 
        Please enter 'Y' or 'N' for questions.
        Args:
        ----
            pickle_file: file storing labeled images to avoid labeling over and over again
        Kwargs:
        ------
            color_space:color space representation you would like
            default:"YUV"
            datatype:str
        """
        assert isinstance(pickle_file,str)
        assert isinstance(color_space,str)
        assert color_space in ['RGB', 'HSV', 'RGB CIE', 'XYZ', 'YUV', 'YIQ', 'YPbPr', 'YCbCr', 'YDbDr'],"Non acceptable colorspace"

        print("Welcome to labeling module.This model will be using roipoly to label pixels as belonging to a class or not.Please enter Y or N for questions",'\n')
        print("Please note default color space is in YUV",'\n')
        storage = Path(pickle_file)
        stored_values = []

        if storage.is_file():
            decision = input("You seem to have started the labeling.Do you wish to continue(Y) or overwrite(N)?")
            if decision=='Y':
                with open(pickle_file, 'rb') as handle:
                    stored_values = pickle.load(handle)

        file_counter = len(stored_values)

        files = self.train_files + self.validation_files

        while(file_counter<self.num_files):
            print(files[file_counter])
            data = {"image":plt.imread(self.root_location + files[file_counter])}
            yuv = color.convert_colorspace(data["image"],"RGB",color_space)
            data["YUV"] = yuv
            data["U"] = (yuv[:,:,1] - np.min(yuv[:,:,1]))/(np.max(yuv[:,:,1]) - np.min(yuv[:,:,1]))
            data["U"] = exposure.adjust_gamma(data["U"],gamma=0.5)
            data["mask"] = np.zeros((yuv.shape[0],yuv.shape[1],len(self.classes)))

            for index,c in enumerate(self.classes[:-1]):
                print("Labeling class:{}".format(c))
                select = True
                while(select):
                    plt.imshow(data["U"])
                    my_roi = RoiPoly(color='r')
                    data["mask"][:,:,index] = data["mask"][:,:,index] + (my_roi.get_mask(data["YUV"][:,:,0])*1)
                    plt.close()
                    select = input("Continue(Y/N)?")=="Y"
                data["mask"][:,:,-1] = data["mask"][:,:,index] + data["mask"][:,:,-1]

            data["mask"][:,:,-1] = np.ones_like(data["mask"][:,:,-1]) - data["mask"][:,:,-1]
            plt.imshow(data["mask"][:,:,-1],cmap='gray')
            plt.show()
            stored_values.append(data)
            decision = input("Do you want to label another image?")
            file_counter = file_counter + 1
            if decision=='N':
                break

        with open(pickle_file, 'wb') as handle:
            pickle.dump(stored_values, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    def Visualize_Labels(self,pickle_file):
        """
        Module to visualize labels acquired from the Labeling module
        Args:
        ----
            self:pointer to current instance of the class
            pickle_file:storage location of data
        """
        with open(pickle_file, 'rb') as handle:
            stored_values = pickle.load(handle) 

        data = random.choice(stored_values)
        plt.subplot(2,2,1)
        plt.imshow(data["image"]),plt.xticks([]),plt.yticks([]),plt.title("Original Image")
        plt.subplot(2,2,2)       
        plt.imshow(data["YUV"][:,:,1],cmap='gray'),plt.xticks([]),plt.yticks([]),plt.title("U")
        plt.subplot(2,2,3)       
        plt.imshow(data["YUV"][:,:,2],cmap='gray'),plt.xticks([]),plt.yticks([]),plt.title("V")
        plt.subplot(2,2,4)       
        plt.imshow(data[self.classes[-1]],cmap='gray'),plt.xticks([]),plt.yticks([]),plt.title("mask")
        plt.show()

    def data_generator(self,pickle_file,window_size=10,step_size=5):
        """
        To generate batches of data for training clorsegmentation models
        Args:
        ----
            pickle_file:
        Kwargs:
        ------
            batch_size:batchsize of data to feed into module
            default :128
            datatype:int
        """

        with open(pickle_file, 'rb') as handle:
            stored_values = pickle.load(handle) 

        length = len(self.train_files)

        for dictionary in stored_values[:length]:
            image = dictionary["YUV"][:,:,1]
            mask = (dictionary["mask"]>0)*1
            for i in range(window_size//2,image.shape[0]-(window_size//2),step_size):
                for j in range(window_size//2,image.shape[1]-(window_size//2),step_size):
                    yield image[i-(window_size//2):i+(window_size//2),j-(window_size//2):j+(window_size//2)],mask[i,j,:]
    


if __name__ == "__main__":
    DM = DataLoader("ECE276A_HW1/trainset/",0.9,["barrel_blue","non_barrel_blue","rest"])
    #DM.display_color_spaces(color_space="YUV")
    DM.Labeling_module("Stored_Values.pickle") #Used for labeling the data.
#DM.Visualize_Labels("Stored_Values.pickle")