'''
ECE276A WI19 HW1
Blue Barrel Detector
'''
import pickle
import os, cv2 ,math
import numpy as np
from skimage import color
from skimage.measure import label, regionprops
from skimage.morphology import disk,square,erosion,dilation,opening,closing
from skimage.feature import match_template

import matplotlib.pyplot as plt 

from LogisticRegression import LogisticRegression

class BarrelDetector():
	def __init__(self,pickle_file="trained_model.pickle"):
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
		with open(pickle_file,"rb") as handle:
			self.model = pickle.load(handle)

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
		grayscale_prediciton = self.model.test_image(img)[:,:,0]
		#predictions =  (grayscale_prediciton[:,:,0] - np.min(grayscale_prediciton[:,:,0])/(np.max(grayscale_prediciton[:,:,0]) - np.min(grayscale_prediciton[:,:,0]))
		grayscale_prediciton = (grayscale_prediciton - np.min(grayscale_prediciton))/(np.max(grayscale_prediciton) - np.min(grayscale_prediciton))
		return grayscale_prediciton>0.5

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
		grayscale_prediciton = self.model.test_image(img)
		selem = square(20)
		opened = opening(grayscale_prediciton,selem)
		label_img = label(opened)
		regions = regionprops(label_img)
		boxes = []
		for props in regions:
			if props.area>50*50 and props.area<500*500 and (props.major_axis_length/props.minor_axis_length)<=2 and (props.major_axis_length/props.minor_axis_length)>=1.25:
				x0,y0 = props.centroid
				orientation = props.orientation
				x1 = x0 + math.cos(orientation) * 0.5 * props.major_axis_length
				y1 = y0 - math.sin(orientation) * 0.5 * props.major_axis_length
				x2 = x0 - math.sin(orientation) * 0.5 * props.minor_axis_length
				y2 = y0 - math.cos(orientation) * 0.5 * props.minor_axis_length
				boxes.append([x1,y1,x2,y2])
		return boxes


if __name__ == '__main__':
	my_detector = BarrelDetector()
	figure_num = 0
	root_location = "ECE276A_HW1/trainset/"
	for file_name in os.listdir(root_location):
		figure_num = figure_num + 1
		fig = plt.figure(figure_num)
		file_name = root_location + file_name
		ax1 = plt.subplot(3,1,1)
		image = cv2.imread(file_name)
		image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
		ax1.imshow(image)
		mask = my_detector.segment_image(image)
		ax2 = plt.subplot(3,1,2)
		ax2.imshow(mask,cmap="gray")
		ax3 = plt.subplot(3,1,3)
		#rect = my_detector.get_bounding_box(image)
		ax3.imshow(image,cmap="gray")
		#ax.plot((x0, x1), (y0, y1), '-r', linewidth=2.5)
		#ax3.add_patch(rect)
		plt.savefig("results/figure{}.png".format(figure_num))
		plt.show()
