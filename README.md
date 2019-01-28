# GMM
This project aims to build an object detection model using Logistic Regression algorithm.

## Programming environment
For the purposes of this project, the model was built using a virtual environment.

    conda create -n myenv python=3.5

<p>Please note that it is essential that version of python is 3.5 as there are cross dependency conflicts that come in due to numpy having depreciated some features with scikit learn.</p>

<p>An external library is used for roi selection :https://github.com/jdoepfert/roipoly.py. Installation instructions are included in the github page.</p>

The remaining libraries are mentioned in requirements.txt. To install these libraries:

    pip install -r requirements.txt

<p>** This is only for training purposes. imageio library is used for visualizing the lableing**</p>


