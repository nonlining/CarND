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
[image5]: ./examples/center.jpg "center image"
[image6]: ./examples/left.jpg "left image"
[image7]: ./examples/right.jpg "right image"

###  Data collecting and Training Strategy

#### 1. Creation of the Training Set & Training Process

I used VGG net as my training model, and the driving skill is also important for this project. If you can drive the car in the middle of road as possible as you can. You can get the better result. But you also have to collect some of data from recovering your car, it also important. This is why collecting road driving data is so important for autonomous car.

#### 2. Appropriate training data

My traning is try to keep car in middle of the road, and run as many as possible, and this simulation will capture 3 images from training lap, cneter image, left image and right image. The following are examples of 3 three images.

![alt text][image5]

![alt text][image6]

![alt text][image7]


### Data Visualization and Preprocessing

First, I must investigate the data for track 1. Data Exploration, Data visualization and Data cleansing are very important for machine learning data preprocessing.

Here is steering angles for track 1.

![alt text][image1]

And Histogram for track 1..

![alt text][image2]

For the above graph. I can know 0 angles take over 70% data. It could make our training overfitting. So I decide to remove some of them. I don't want to adjust my data to strict normal distrubtation. Since it also reflect the true for this track. So I just remove 40% of them.

![alt text][image3]

Also , the above image is after augment data with flip images.

![alt text][image4]


### Model Architecture

#### 1. An appropriate model architecture has been employed


| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 RGB image   							| 
| cropping2d         		| 65x320x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, outputs 63*318*32 	|
| RELU					|												|
| Max pooling	2*2       	| 2x2 stride,  outputs 31*159*32 				|
| Dropout layers 		| 30% dropouts       									|
| Convolution 3x3	    | 1x1 stride, outputs 29*157x64  |
| RELU					|												|
| Max pooling	2*2       	| 2x2 stride,  outputs 14*78*64 				|
| Dropout layers 		| 30% dropouts       									|
| Convolution 3x3	    | 1x1 stride, outputs 12*76*128  |
| RELU					|												|
| Max pooling	2*2       	| 2x2 stride,  outputs 6*38*256 				|
| Convolution 3x3	    | 1x1 stride, outputs 4*36*256  |
| RELU					|												|
| Max pooling	2*2       	| 2x2 stride,  outputs 2*18*256 				|
| Fully connected		| 128 Inputs        									|
| Fully connected		| 64 Inputs        									|
| Fully connected		| 32 Inputs        									|
| Fully connected		| 1 Inputs        									|


#### 2. Attempts to reduce overfitting in the model

Beacase there are many 0 angles data, I add dropout to 30% to avoid overfitting. I also run a another run for vaildation, so my vaildation data is a whole track data.

#### 3. Model parameter tuning

How to determine the deep of model and how much dropout I should use is the first step. I use 3*3 kernel size for very Convolution layers and 2*2 size for max pooling. At the first, I only use 10% for dropout layers, and the result is not so good. So I changed this to 30% dropout.

#### 4. Result for lap 1
[![video](https://github.com/nonlining/CarND/blob/master/CarND-Behavioral-Cloning-P3/2017_08_15_15_21_04_419.jpg)](https://youtu.be/D-e1tCMtSYc)

