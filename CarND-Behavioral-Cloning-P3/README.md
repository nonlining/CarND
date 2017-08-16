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

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
lambda_5 (Lambda)                (None, 160, 320, 3)   0           lambda_input_5[0][0]             
____________________________________________________________________________________________________
cropping2d_5 (Cropping2D)        (None, 65, 320, 3)    0           lambda_5[0][0]                   
____________________________________________________________________________________________________
convolution2d_17 (Convolution2D) (None, 63, 318, 32)   896         cropping2d_5[0][0]               
____________________________________________________________________________________________________
maxpooling2d_17 (MaxPooling2D)   (None, 31, 159, 32)   0           convolution2d_17[0][0]           
____________________________________________________________________________________________________
dropout_9 (Dropout)              (None, 31, 159, 32)   0           maxpooling2d_17[0][0]            
____________________________________________________________________________________________________
convolution2d_18 (Convolution2D) (None, 29, 157, 64)   18496       dropout_9[0][0]                  
____________________________________________________________________________________________________
maxpooling2d_18 (MaxPooling2D)   (None, 14, 78, 64)    0           convolution2d_18[0][0]           
____________________________________________________________________________________________________
dropout_10 (Dropout)             (None, 14, 78, 64)    0           maxpooling2d_18[0][0]            
____________________________________________________________________________________________________
convolution2d_19 (Convolution2D) (None, 12, 76, 128)   73856       dropout_10[0][0]                 
____________________________________________________________________________________________________
maxpooling2d_19 (MaxPooling2D)   (None, 6, 38, 128)    0           convolution2d_19[0][0]           
____________________________________________________________________________________________________
convolution2d_20 (Convolution2D) (None, 4, 36, 256)    295168      maxpooling2d_19[0][0]            
____________________________________________________________________________________________________
maxpooling2d_20 (MaxPooling2D)   (None, 2, 18, 256)    0           convolution2d_20[0][0]           
____________________________________________________________________________________________________
flatten_5 (Flatten)              (None, 9216)          0           maxpooling2d_20[0][0]            
____________________________________________________________________________________________________
dense_13 (Dense)                 (None, 128)           1179776     flatten_5[0][0]                  
____________________________________________________________________________________________________
dense_14 (Dense)                 (None, 64)            8256        dense_13[0][0]                   
____________________________________________________________________________________________________
dense_15 (Dense)                 (None, 1)             65          dense_14[0][0]                   
====================================================================================================
Total params: 1,576,513
Trainable params: 1,576,513
Non-trainable params: 0


#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

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
