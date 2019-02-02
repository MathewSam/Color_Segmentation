# Color Segementation
<p>This project aims to build an object detection model using Logistic Regression algorithm. The provided dataset consists of scenery with blue drums in it. The objective is to build a color segementation model using logistic regression to "detect" pixels with high probability of being part of a barrel.</p>

## Approach
### Preprocessing
<p>Not a lot of preprocessing was required since the barrels in the training set were placed in environments without a lot of other blue objects in the scene. To avoid confusion between the barrels and the white spaces in the image, the images where transfered to the YUV space. The U component is basically B-G from the RGB space. This allows filtering out white spaces from the data and only purely blue objects are left behind</p>

### Acquiring training data
<p>The training data is acquired using a windowing process that took wxw windows from images and fed it into the model. A generator was used to sequentially provide the data to the model.</p>

### Training procedure
<p>The model was trained using stochastic gradient descent with parameter updates with every element. Stochastic gradient descent is more robust that batch or mini batch gradient descent since the model enconters a lot of noise. This randomness in the gradient update step prevents the model from falling into local minima requiring fewer iterations</p>

The current model was trained with a learning rate of 0.01 for 30 epochs using about 90% of the training data provided 

## Files
## DataLoader.py
This file contains the code used to label and store the training and validation data. The functional modules used are :
### data_generator
To generate batches of data for training clorsegmentation models
### Visualize_Labels
Module to visualize labels acquired from the Labeling module
### Labeling_module
Module to label data in consideration.

All other functions in this file were built for debugging purposes.

## barrel_detector.py
This file contains the code used to label and store the training and validation data. The functional modules used are :
### train
Module takes up a sample generator(acquired from DataLoader) and uses samples to carry out gradient descent using stochastic gradient descent.
### test_image
Module tests input image using sliding windows to predict if center pixel is of barrel class or not.
### segement_image
Module used to segement image to get segementation masks
### get_bounding_box
Find the bounding box of the blue barrel

## Programming environment
For the purposes of this project, the model was built using a virtual environment.

    conda create -n myenv python=3.5

<p>Please note that it is essential that version of python is 3.5 as there are cross dependency conflicts that come in due to numpy having depreciated some features with scikit learn.</p>

<p>An external library is used for roi selection :https://github.com/jdoepfert/roipoly.py. Installation instructions are included in the github page.</p>

The remaining libraries are mentioned in requirements.txt. To install these libraries:

    pip install -r requirements.txt

<p>** This is only for training purposes. imageio library is used for visualizing the lableing**</p>


