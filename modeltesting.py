"""
This file aims to battle test the model on that validation set. Happy coding!!! 
"""
__author__ = "Mathew Sam"
__version__ = "1.0.0"
__email__ = "m1sam@eng.ucsd.edu"
__maintainer__ = "Mathew Sam"

from LogisticRegression import LogisticRegression
from DataLoader import DataLoader

import numpy as np 
from scipy import ndimage
import matplotlib.pyplot as plt
import pickle

from skimage.morphology import disk,square,erosion,dilation
from skimage import exposure,color
import cv2


DM = DataLoader("ECE276A_HW1/trainset/",0.8,["barrel_blue","non_barrel_blue","rest"])
pickle_file = "trained_models/model1.pickle"



with open(pickle_file, 'rb') as handle:
    model = pickle.load(handle) 

figure_num =0
for file_name in DM.train_files:
    figure_num = figure_num + 1
    plt.figure(figure_num)
    file_name = DM.root_location + file_name
    plt.subplot(2,1,1)
    image = plt.imread(file_name)
    plt.imshow(image),plt.xticks([]),plt.yticks([]),plt.title("Original image")
    mask = model.test_image(image)
    plt.subplot(2,1,2)
    mask = mask[:,:,0]>0.7
    selem = disk(10)
    mask = dilation(mask, selem)
    plt.imshow(mask,cmap="gray"),plt.xticks([]),plt.yticks([]),plt.title("Class1")
    plt.savefig("results/figure{}.png".format(figure_num))
    plt.show()