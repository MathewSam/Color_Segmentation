'''
ECE276A WI19 HW1
Blue Barrel Detector
'''
import pickle
import os, cv2 ,math
import numpy as np
from skimage import color
from skimage.filters import threshold_otsu, rank
from skimage.measure import label, regionprops
from skimage.morphology import disk,square,erosion,dilation,opening,closing
from skimage.feature import match_template

import matplotlib.pyplot as plt 

class BarrelDetector():
	def __init__(self,window_size=10,num_classes=2,Train=False):
		"""
		Initializes the barrel detector module to detect barrels in the image. decides if model need to be trained again from scratch
		using the Train keyword argument
		Args:
		---
			self:pointer to current instance of the class
		Kwargs:
		------
			window_size: size of window for training
			default: 10
			datatype: int

			num_classes: number of classes to train the model for
			default: 2
			datatype:int

			Train:boolean value to decide whether the model needs to be trained again from scratch
			default:False
			datatype:bool
		"""
		assert isinstance(window_size,int),"Please enter an integer value for window size"
		assert isinstance(num_classes,int),"Number of classes must be an integer"
		assert isinstance(Train,bool),"Train must be a boolean value"
		if Train:
			self.num_classes = num_classes
			self.W = np.random.rand(window_size**2,num_classes)
			self.b = np.random.rand(1,num_classes)
			self.window_size = window_size
		else:
			self.num_classes = 2
			self.window_size = window_size
			self.W = np.array([[ 1.71697673 ,-1.26367697],
 [ 1.53335629 ,-1.06336256],
 [ 1.24487625 ,-1.08333945],
 [ 1.54904934 ,-1.0956119 ],
 [ 1.55892478 ,-1.23564843],
 [ 1.38412832 ,-0.67680309],
 [ 1.6579981  ,-0.77127117],
 [ 1.6909461  ,-1.27405679],
 [ 2.03187581 ,-1.72310603],
 [ 2.35370275 ,-1.93722097],
 [ 1.75454258 ,-0.82782152],
 [ 0.77021436 ,-0.67179031],
 [ 0.92465042 ,-0.30089412],
 [ 1.23336417 ,-0.25744085],
 [ 0.98063547 ,-0.52691043],
 [ 0.81014663 , 0.06817363],
 [ 0.87574209 , 0.36726683],
 [ 1.34375976 , 0.10428078],
 [ 1.12919707 , 0.13266028],
 [ 1.94826146 ,-1.01596313],
 [ 1.10632223 ,-0.10140011],
 [ 1.00446984 , 0.29380051],
 [ 0.83732022 , 0.39494687],
 [ 0.53037033 , 0.0319112 ],
 [ 0.68414985 ,-0.10640564],
 [ 0.98203027 ,-0.17131617],
 [ 0.96460972 , 0.60410879],
 [ 0.39237254 , 0.31252097],
 [ 0.81719994 , 0.64980304],
 [ 1.66637991 ,-1.25334716],
 [ 0.79893014 ,-0.17360642],
 [ 1.06446069 ,-0.04402954],
 [ 0.66302694 , 0.26491216],
 [ 0.7958242  , 0.14906986],
 [ 0.77573279 , 0.40450295],
 [ 0.26761164 , 0.33788829],
 [ 0.20201225 , 0.89370697],
 [ 0.68344424 , 0.45161353],
 [ 0.62592197 , 0.8894093 ],
 [ 1.73491182 ,-0.49106942],
 [ 0.69076839 , 0.15545562],
 [ 0.75083963 , 0.0171033 ],
 [ 0.61274519 , 0.62658617],
 [ 0.72120623 ,-0.08837089],
 [ 0.85184926 ,-0.23604947],
 [ 0.19772576 , 0.4622179 ],
 [-0.02960906 , 0.71122582],
 [ 0.35635143 , 1.12691252],
 [ 0.39565686 , 0.89103127],
 [ 1.19643168 ,-0.39392094],
 [ 1.06126468 , 0.08203286],
 [ 1.24130796 ,-0.2980852 ],
 [ 0.30255355 , 0.02634161],
 [ 0.29149014 , 0.32255676],
 [ 0.79892838 , 0.56154803],
 [ 0.501266   , 0.85890083],
 [ 0.29436923 , 1.31271087],
 [ 0.70170359 , 0.87895055],
 [ 0.94063152 , 1.10751936],
 [ 1.58451986 ,-0.7707969 ],
 [ 0.62698649 , 0.29412583],
 [ 0.65526565 , 0.30566674],
 [ 0.33711689 , 0.14033831],
 [ 0.89159217 , 0.32464274],
 [ 0.61785468 , 0.37036704],
 [ 0.14824654 , 0.8367593 ],
 [ 0.03071353 , 0.95639791],
 [ 0.54188744 , 1.36373742],
 [ 0.27974134 , 0.43211861],
 [ 1.66000249 ,-0.05691579],
 [ 0.98447143 ,-0.84131494],
 [ 1.46988228 ,-0.81022662],
 [ 0.87497531 ,-0.60459929],
 [ 0.65813469 , 0.32990532],
 [ 1.3491013  , 0.51268404],
 [ 0.79005878 , 0.61019497],
 [ 0.83659299 , 0.41162929],
 [ 0.5792906  , 0.8129943 ],
 [ 0.97238105 , 0.65973444],
 [ 1.71190788 ,-1.0509217 ],
 [ 1.25570706 ,-0.59492676],
 [ 1.5624916  ,-0.86986423],
 [ 1.49512207 ,-0.18945951],
 [ 1.2757876  ,-0.8994241 ],
 [ 0.78671523 , 0.04942296],
 [ 1.3625909  ,-0.47625877],
 [ 0.87245698 , 0.57273157],
 [ 0.40946194 , 0.59548379],
 [ 1.07507595 ,-0.13287104],
 [ 1.87179426 ,-1.09221575],
 [ 2.07773138 ,-1.6961334 ],
 [ 2.41335822 ,-1.05547576],
 [ 1.4703667  ,-1.18314337],
 [ 1.35493527 ,-1.3549783 ],
 [ 1.52976    ,-1.10147442],
 [ 1.37046998 ,-1.00713041],
 [ 0.97633099 ,-0.66359694],
 [ 1.20821827 ,-0.26771501],
 [ 1.71070577 ,-0.57049583],
 [ 1.85996464 ,-1.51259867]])
		self.b = np.array([[-1.96700324 , 3.53692749]])

	
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
		
		error_plot = [0]
		for _ in range(epochs):
			loss = 0
			for sample,label in sample_generator:
				y = self.sigmoid(sample.reshape(1,-1),self.W,self.b)
				gradient = label - y
				gradient = gradient*np.array([1,2])
				self.W = self.W + learning_rate*gradient*sample.reshape(-1,1)
				self.b = self.b + learning_rate*gradient
				CE = np.mean(-label*np.log(y+epsilon) - (1-label)*np.log(1-y+epsilon))
				loss = (loss*0.9) + (CE*0.1)
			error_plot.append(loss)
			learning_rate = learning_rate*0.99

		plt.plot(error_plot,label="Training error")
		plt.grid()
		plt.xlabel("epochs")
		plt.ylabel("Cross Entropy loss")
		plt.legend()
		plt.show()
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
		img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
		grayscale_prediciton = self.test_image(img)[:,:,0]
		#predictions =  (grayscale_prediciton[:,:,0] - np.min(grayscale_prediciton[:,:,0])/(np.max(grayscale_prediciton[:,:,0]) - np.min(grayscale_prediciton[:,:,0]))
		grayscale_prediciton = (grayscale_prediciton - np.min(grayscale_prediciton))/(np.max(grayscale_prediciton) - np.min(grayscale_prediciton))
		
		#thresh = threshold_otsu(grayscale_prediciton)
		grayscale_prediciton =grayscale_prediciton>=0.8
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
		img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
		grayscale_prediction = self.test_image(img)[:,:,0]
		grayscale_prediction = (grayscale_prediction - np.min(grayscale_prediction))/(np.max(grayscale_prediction) - np.min(grayscale_prediction))
		
		#if(np.mean(grayscale_prediction)>0.1):
		#	grayscale_prediction =grayscale_prediction>=0.2
		#else:
		#	grayscale_prediction = grayscale_prediction>=0.9
		if(np.mean(grayscale_prediction)<0.1):
			thresh = threshold_otsu(grayscale_prediction)
			grayscale_prediction =grayscale_prediction>=thresh
		else:
			grayscale_prediction =grayscale_prediction>=0.8

		selem = disk(15)
		opened = closing(grayscale_prediction,selem)

		label_img = label(opened)
		regions = regionprops(label_img,coordinates='xy')
		boxes = []
		for props in regions:
			y0, x0 = props.centroid
			if props.minor_axis_length >0 and props.major_axis_length/props.minor_axis_length<3 and abs(180*props.orientation/math.pi)>75 and props.area>1000:
				#print("Ration : {}".format(180*props.orientation/math.pi))
				ymin = int(y0 - (props.major_axis_length/2))
				ymax = int(y0 + (props.major_axis_length/2))
				xmax = int(x0 + (props.minor_axis_length/2))
				xmin = int(x0 - (props.minor_axis_length/2))
				boxes.append([xmin,ymin,xmax,ymax])
				print((xmin,xmax,ymin,ymax))
		print('\n')
		return boxes


if __name__ == '__main__':
	from DataLoader import DataLoader
	my_detector = BarrelDetector(window_size=10,num_classes=2,Train=False)
	#train_data_root="ECE276A_HW1/trainset/"
	#train_data_split=0.9
	#classes = ["barrel_blue","rest"]
	#DM = DataLoader(train_data_root,train_data_split,classes)
	#pickle_file="Stored_Values2.pickle"
	#gen = DM.data_generator(pickle_file,window_size=10,step_size=5)
	#my_detector.train(gen,epochs=30,learning_rate=0.01)
	
	#figure_num = 0
	root_location = "ECE276A_HW1/trainset/"
	for file_name in os.listdir(root_location):
		plt.figure(1)
		file_name = root_location + file_name
		image = cv2.imread(file_name)
		img = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
		plt.imshow(img)
		plt.figure(2)
		mask = my_detector.segment_image(image)
		plt.imshow(mask,cmap="gray")
		plt.figure(3)
		rect = my_detector.get_bounding_box(image)
		plt.imshow(img)
		for c in rect:
			minc, minr, maxc, maxr = c
			bx = (minc, maxc, maxc, minc, minc)
			by = (minr, minr, maxr, maxr, minr)
			plt.plot(bx, by, '-r', linewidth=2.5)
		#plt.savefig("results/figure{}.png".format(figure_num))
		plt.show()
		
