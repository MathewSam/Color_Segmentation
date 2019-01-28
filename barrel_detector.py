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

class BarrelDetector():
	def __init__(self,window_size=10,num_classes=2,Train=False,pickle_file="trained_model.pickle"):
		"""
		"""
		if Train:
			self.num_classes = num_classes
			self.W = np.random.rand(window_size**2,num_classes)
			self.b = np.random.rand(1,num_classes)
			self.window_size = window_size
		else:
			self.num_classes = 2
			self.window_size = window_size
			self.W =  np.array([[ 1.44429789 ,-0.71831929],[ 1.23343677, -0.46352353],[ 0.94217466 ,-0.47793628],[ 1.23097975 ,-0.45947272]
 					,[ 1.24638665 ,-0.61057217],[ 1.13387323 ,-0.1762929 ],[ 1.49716415 ,-0.44960327],[ 1.58658584 ,-1.06533627],[ 1.79257469 ,-1.24450378],[ 1.71664372 ,-0.66310291]
 					,[ 1.65677011 ,-0.63227658],[ 0.63724715 ,-0.40585589],[ 0.80569214 ,-0.06297755],[ 1.12562776 ,-0.04196804],[ 0.90435214 ,-0.37434377],[ 0.7980025  , 0.09246189]
 					,[ 0.95486723 , 0.20901656],[ 1.50319529 ,-0.21459028],[ 1.18314381 , 0.02476681],[ 1.6366031  ,-0.39264642],[ 1.06434469 ,-0.01744503],[ 0.89855076 , 0.50563868]
 					,[ 0.73937198 , 0.59084335],[ 0.4466759  , 0.19930006],[ 0.65474798 ,-0.0476019 ],[ 1.05939828 ,-0.3260522 ],[ 1.14702789 , 0.23927245],[ 0.67941532 ,-0.26156459]
 					,[ 1.00318722 , 0.27782848],[ 1.49982828 ,-0.9202439 ],[ 0.80000177 ,-0.17574968],[ 0.98983021 , 0.10523142],[ 0.58774994 , 0.41546616],[ 0.7061369  , 0.32844446]
 					,[ 0.73979775 , 0.47637302],[ 0.37184392 , 0.12942374],[ 0.4285918  , 0.44054787],[ 1.00333577 ,-0.18816953],[ 0.85126254 , 0.43872816],[ 1.6228581  ,-0.26696199]
 					,[ 0.66291248 , 0.21116744],[ 0.69991691 , 0.11894874],[ 0.55282693 , 0.74642268],[ 0.60562076 , 0.14280004],[ 0.79386927 ,-0.12008949],[ 0.28831534 , 0.28103874]
 					,[ 0.20615625 , 0.23969521],[ 0.69319358 , 0.45322822],[ 0.61487954 , 0.45258591],[ 1.09628772 ,-0.19363301],[ 1.00212606 , 0.2003101 ],[ 1.19278385 ,-0.20103697]
 					,[ 0.23382989 , 0.16378893],[ 0.19129496 , 0.52294712],[ 0.74781421 , 0.66377636],[ 0.58840831 , 0.68461621],[ 0.50946408 , 0.88252117],[ 1.01266149 , 0.25703474]
 ,[ 1.14848683 , 0.69180874]
 ,[ 1.47196995 ,-0.54569708]
 ,[ 0.56156993 , 0.42495895]
 ,[ 0.60169816 , 0.41280172]
 ,[ 0.3081048  , 0.19836249]
 ,[ 0.86397156 , 0.37988395]
 ,[ 0.62859542 , 0.34888556]
 ,[ 0.26390523 , 0.60544192]
 ,[ 0.25587942 , 0.50606613]
 ,[ 0.88232025 , 0.68287181]
 ,[ 0.5243625  ,-0.0571237 ]
 ,[ 1.56184381 , 0.13940157]
 ,[ 0.80970072 ,-0.49177352]
 ,[ 1.28009968 ,-0.43066141]
 ,[ 0.72776828 ,-0.31018522]
 ,[ 0.513788   , 0.61859869]
 ,[ 1.28527962 , 0.6403274 ]
 ,[ 0.84280775 , 0.50469703]
 ,[ 0.97174832 , 0.14131863]
 ,[ 0.80016828 , 0.37123895]
 ,[ 1.07513963 , 0.45421728]
 ,[ 1.46590889 ,-0.55892372]
 ,[ 0.90758035 , 0.10132666]
 ,[ 1.21858775 ,-0.18205653]
 ,[ 1.20395154 , 0.39288155]
 ,[ 1.00774955 ,-0.363348  ]
 ,[ 0.60816791 , 0.4065176 ]
 ,[ 1.32023878 ,-0.39155454]
 ,[ 0.93562856 , 0.44638841]
 ,[ 0.54791427 , 0.31857913],[ 1.05713969 ,-0.09699852],[ 1.47573752 ,-0.30010227],[ 1.45869406 ,-0.45805875],[ 1.81044942 , 0.15034184]
 ,[ 0.9063961  ,-0.05520217],[ 0.82589407 ,-0.2968959 ],[ 1.12546753 ,-0.29288948],[ 1.11255482 ,-0.49130009],[ 0.83886659 ,-0.38866815],[ 1.12491984 ,-0.10111816],[ 1.47264762 ,-0.09437953],[ 1.25180483 ,-0.29627904]])
		self.b = np.array([[-1.81666353 , 3.23624807]])

	
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

	def train(self,sample_generator,epochs=5,learning_rate=0.001,epsilon = 0.00000001,pickle_file="trained_model.pickle"):
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

		for epoch in range(epochs):
			for sample,label in sample_generator:
				y = self.sigmoid(sample.reshape(1,-1),self.W,self.b)
				gradient = label - y
				self.W = self.W + learning_rate*gradient*sample.reshape(-1,1)
				self.b = self.b + learning_rate*gradient
			learning_rate = learning_rate*0.99

		print("Weight matrix = \n {}\n".format(self.W))
		print("bias matrix = \n {}\n".format(self.b))
	def test_image(self,input_image):
		"""
		Testing images using logistoic regression
		Args:
		----
			self:pointer to current instance of the class
			input_image:input test image
		"""
		mask = np.zeros((input_image.shape[0],input_image.shape[1],self.num_classes))
		image = color.convert_colorspace(input_image,"RGB","YUV")[:,:,1]
		for i in range(self.window_size//2,mask.shape[0]-(self.window_size//2)):
			for j in range(self.window_size//2,mask.shape[1]-(self.window_size//2)):
				window = image[i-(self.window_size//2):i+(self.window_size//2),j-(self.window_size//2):j+(self.window_size//2)]
				mask[i,j,:] = self.sigmoid(window.reshape(1,-1),self.W,self.b)
		return mask

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
		grayscale_prediciton = self.test_image(img)[:,:,0]
		#predictions =  (grayscale_prediciton[:,:,0] - np.min(grayscale_prediciton[:,:,0])/(np.max(grayscale_prediciton[:,:,0]) - np.min(grayscale_prediciton[:,:,0]))
		grayscale_prediciton = (grayscale_prediciton - np.min(grayscale_prediciton))/(np.max(grayscale_prediciton) - np.min(grayscale_prediciton))
		return grayscale_prediciton

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
		grayscale_prediciton = self.test_image(img)
		selem = square(20)
		opened = opening(grayscale_prediciton,selem)
		label_img = label(opened)
		regions = regionprops(label_img)
		boxes = []
		for props in regions:
			if props.area>50*50 and props.area<500*500 and (props.major_axis_length/props.minor_axis_length)<=2 and (props.major_axis_length/props.minor_axis_length)>=1.25:
				x0,y0 = props.centroid
				orientation = props.orientation
				x1 = x0 + (math.cos(orientation) * 0.5 * props.major_axis_length)
				y1 = y0 - (math.sin(orientation) * 0.5 * props.major_axis_length)
				x2 = x0 - (math.sin(orientation) * 0.5 * props.minor_axis_length)
				y2 = y0 - (math.cos(orientation) * 0.5 * props.minor_axis_length)
				boxes.append([x2,y2,x1,y1])
		return boxes


if __name__ == '__main__':
	from DataLoader import DataLoader
	my_detector = BarrelDetector(window_size=10,num_classes=2,Train=False)
#	train_data_root="ECE276A_HW1/trainset/"
#	train_data_split=0.9
#	classes = ["barrel_blue","rest"]
#	DM = DataLoader(train_data_root,train_data_split,classes)
#	pickle_file="Stored_Values2.pickle"
#	gen = DM.data_generator(pickle_file,window_size=10,step_size=2)
#	my_detector.train(gen,epochs=30,learning_rate=0.01)
	
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
