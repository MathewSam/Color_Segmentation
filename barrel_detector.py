'''
ECE276A WI19 HW1
Blue Barrel Detector
'''

import os, cv2
from skimage.measure import label, regionprops
from skimage.morphology import disk,square,erosion,dilation

import matplotlib.pyplot as plt 

from DataLoader import DataLoader
from LogisticRegression import LogisticRegression

class BarrelDetector():
	def __init__(self,window_size,DM,classes,pickle_file="labeled_data/Stored_Values.pickle"):
		'''
			Initilize your blue barrel detector with the attributes you need
			eg. parameters of your classifier
			Args:
			----
				self:pointer to current instance of the class
				window_size: size of window used for training
				DM:data manager built from DataLoader class
				classes:classes the data has been factored into
			Kwargs:
			------
				pickle_file:location of pickle file with data labeled using dataloader's labeling module
				default:"labeled_data/Stored_Values.pickle"
		'''
		
		self.model = LogisticRegression(window_size,num_classes=len(classes))
		gen = DM.data_generator(pickle_file,window_size=window_size,step_size=2)
		self.model.train(gen,epochs=1000,learning_rate=0.1)

	def segment_image(self, img):
		'''
			Calculate the segmented image using a classifier
			eg. Single Gaussian, Gaussian Mixture, or Logistic Regression
			call other functions in this class if needed
			
			Inputs:
				img - original image
			Outputs:
				mask_img - a binary image with 1 if the pixel in the original image is blue and 0 otherwise
		'''
		# YOUR CODE HERE
		grayscale_prediciton = self.model.test_image(img)
		mask_img = grayscale_prediciton[:,:,0]>0.99
		return mask_img

	def get_bounding_box(self, img):
		'''
			Find the bounding box of the blue barrel
			call other functions in this class if needed
			
			Inputs:
				img - original image
			Outputs:
				boxes - a list of lists of bounding boxes. Each nested list is a bounding box in the form of [x1, y1, x2, y2] 
				where (x1, y1) and (x2, y2) are the top left and bottom right coordinate respectively. The order of bounding boxes in the list
				is from left to right in the image.
				
			Our solution uses xy-coordinate instead of rc-coordinate. More information: http://scikit-image.org/docs/dev/user_guide/numpy_images.html#coordinate-conventions
		'''
		# YOUR CODE HERE
		grayscale_prediciton = self.model.test_image(img)[:,:,0]
		return grayscale_prediciton


if __name__ == '__main__':
	train_data_root="ECE276A_HW1/trainset/"
	train_data_split=0.9
	classes = ["barrel_blue","non_barrel_blue","rest"]
	DM = DataLoader(train_data_root,train_data_split,classes)
	my_detector = BarrelDetector(10,DM,classes)
	figure_num = 0
	for file_name in DM.train_files:
		figure_num = figure_num + 1
		plt.figure(figure_num)
		file_name = DM.root_location + file_name
		plt.subplot(3,1,1)
		image = plt.imread(file_name)
		plt.imshow(image),plt.xticks([]),plt.yticks([]),plt.title("Original image")
		mask = my_detector.segment_image(image)
		plt.subplot(3,1,2)
		plt.imshow(mask,cmap="gray"),plt.xticks([]),plt.yticks([]),plt.title("Class1")
		plt.subplot(3,1,3)
		gray_scale = my_detector.get_bounding_box(image)
		plt.imshow(gray_scale,cmap="gray"),plt.xticks([]),plt.yticks([]),plt.title("Class1")
		plt.show()

