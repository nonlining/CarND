# Traffic Sign Recognition
---
**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./SignImageDistribu.png "Visualization"
[image2]: ./SignImageRandomPick.png "SignImages43Classes"
[image3]: ./beforeNormalization.png "beforeNorm"
[image4]: ./afterNormalization.png "AfterNorm"
[image5]: ./test_data/bumpy-road.jpg "Traffic Sign 1"
[image6]: ./test_data/speed_limit.jpg "Traffic Sign 2"
[image7]: ./test_data/stop.jpg "Traffic Sign 3"
[image8]: ./test_data/traffic-sign-1443060__180.jpg "Traffic Sign 4"
[image9]: ./test_data/traffic-signs-achtung-unfallschwerpunkt-german-for-warning-accident-CRDR2P.jpg "Traffic Sign 5"
[image10]: ./bar1.png "Traffic Sign 1"
[image11]: ./bar2.png "Traffic Sign 2"
[image12]: ./bar3.png "Traffic Sign 3"
[image13]: ./bar4.png "Traffic Sign 4"
[image14]: ./bar5.png "Traffic Sign 5"

---
### README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/nonlining/CarND/tree/master/CarND-Traffic-Sign-Classifier-Project)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is ? 34799
* The size of the validation set is ? 4410
* The size of test set is ? 12630
* The shape of a traffic sign image is ? (32, 32, 3)
* The number of unique classes/labels in the data set is ? 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data distribution. 

![image_distribution][image1]

Then, this is the random pick images from 43 classes. It tells you how these signs look like.

![SignImages][image2]

### Design and Test a Model Architecture

#### 1.Data pre-processing

I only use normalization to all images, and transfer label data to one-hot-encoding. I decide not to transfer images to grayscale, because I would like to observe that RGB image will impact prediction accuracy. 

Here is an example of a traffic sign image before and after normalization.

![alt text][image3]

![alt text][image4]

Basically, it's no difference for human eye.


#### 2.Final model architecture.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, VALID padding, outputs 28x28x30 	|
| RELU					|												|
| Max pooling	2*2       	| 2x2 stride,  outputs 16x16x30 				|
| Convolution 5x5	    | 1x1 stride, VALID padding, outputs 10x10x60  |
| RELU					|												|
| Max pooling	2*2       	| 2x2 stride,  outputs 5x5x60 				|
| Convolution 3x3	    | 1x1 stride, VALID padding, outputs 3x3x120  |
| Fully connected		| 1080 Inputs        									|
| Fully connected		| 120 Inputs        									|
| Dropout layers 		| 50& dropouts       									|
| Softmax				| 43 classes        									|


#### 3. Training Model.

The HyperParameters that I choiced as following.
rate = 0.001
epochs = 100
batch_size = 128
keep_probability = 0.5

I choice AdamOptimizer for my optimizer.

#### 4. Tuning Model.

At first, I used 2 layers convolutions in my model, but the validation set of accuracy always limit to around 0.75, and I tried to add one more convolutions in model. This more complexity model brings 0.80 accuracy for validation set. It was far from 0.94. So, I took a lot of time on tuning other hyperparameters, but these did not help.

Finally, I tried to adjust my normalization function. This modified it to (x - 0/255.0). This modification boost the accuracy to 0.90. I also add depths of my convolution layers. This could be too over complexity and could be lead overfitting, so I add dropout layer for this.

Here is my final model results were:
* training set accuracy of 0.998
* validation set accuracy of 0.960
* test set accuracy of 0.950

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image5] ![alt text][image6] ![alt text][image7] 
![alt text][image8] ![alt text][image9]

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Bummy Road     		| Bummy Road   									| 
| Speed Limit 30km/h     			| Speed Limit 30km/h 										|
| Stop					| Stop											|
| Road Work	      		| Road Work					 				|
| General Caution			| General Caution      							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. 
This compares the whole test set accuracy is 95 %

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. 

![alt text][image10] ![alt text][image11] ![alt text][image12] 
![alt text][image13] ![alt text][image14]

Obviously, these 5 images all easy to prediction. So the highest probability of these images all near to 1.0.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


