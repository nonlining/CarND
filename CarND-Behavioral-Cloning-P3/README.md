# Behavioral Cloning

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./Data_visualization.png "Model Visualization"
[image2]: ./Histogram1.png "Histogram1"
[image3]: ./Histogram2.png "Histogram2"
[image4]: ./Histogram3.png "Histogram3"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

### Data Visualization

First, I must investigate the data for track 1. Data Exploration, Data visualization and Data cleansing are very important for machine learning data preprocessing.

Here is steering angles for track 1.

![alt text][image1]

And Histogram for track 1..

![alt text][image2]

For the above graph. I can know 0 angles take over 70% data. It could make our training overfitting. So I decide to remove some of them.


### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed


#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning



#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture


| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 RGB image   							| 
| cropping2d         		| 65x320x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, outputs 63*318*32 	|
| RELU					|												|
| Max pooling	2*2       	| 2x2 stride,  outputs 31*159*32 				|
| Dropout layers 		| 30& dropouts       									|
| Convolution 3x3	    | 1x1 stride, outputs 29*157x64  |
| RELU					|												|
| Max pooling	2*2       	| 2x2 stride,  outputs 14*78*64 				|
| Dropout layers 		| 30& dropouts       									|
| Convolution 3x3	    | 1x1 stride, outputs 12*76*128  |
| RELU					|												|
| Max pooling	2*2       	| 2x2 stride,  outputs 6*38*256 				|
| Convolution 3x3	    | 1x1 stride, outputs 4*36*256  |
| RELU					|												|
| Max pooling	2*2       	| 2x2 stride,  outputs 2*18*256 				|
| Fully connected		| 128 Inputs        									|
| Fully connected		| 64 Inputs        									|
| Fully connected		| 64 Inputs        									|
| Fully connected		| 1 Inputs        									|


The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
