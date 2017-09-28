**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)

[image1]: ./Data_Visualization.png "image1"
[image2]: ./extract_feature_car.png "image2"
[image3]: ./extract_feature_noncar.png "image3"
[image4]: ./test_image_result.png "image4"
[image5]: ./heat_map_6.png "image5"
[image6]: ./bounding_boxes.png "image6"
[image7]: ./bounding_boxes.png "image6"
[image8]: ./fit_poly.png "image8"
[image10]: ./draw_lane_info.png "image10"
[video1]: ./project4_adv_lane_video.mp4 "Video"
[video2]: ./project4_challenge_adv_lane_video.mp4 "Video"

---

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell of the IPython notebook located in [Project5.ipynb](https://github.com/nonlining/CarND/edit/master/CarND-Vehicle-Detection/Project5.ipynb).  

The first step of this project is data visualization to know what data look like.

![alt text][image1]

The following images are extracted HOG image from different color spaces and channels.

![alt text][image2]
![alt text][image3]

#### 2. Explain how you settled on your final choice of HOG parameters.

I used a python script to run over all parameters combination to find out the best result(Accuracy). This is brute force way to get the best result. The following is my all parameters for this code. [getParam.py](https://github.com/nonlining/CarND/blob/master/CarND-Vehicle-Detection/getParam.py).  


```python
color_spaces = ['HSV','LUV', 'HLS', 'YUV', 'YCrCb', 'RGB']
orients = [5, 6, 7, 8, 9, 10]
pix_per_cells = [8,9,10,11,12]
cell_per_blocks = [2,3,4,5,6]
````
It tooks 6*6*5*5 = 900 iterations and 15 hours to complete the parameters.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM with the parameters that I got from getParam.py. The parameters are as following:

```python
color_space = 'HLS' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 8  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 3 # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
```

But, This result on test images and video is very bad. There are many false true detections, and bounding boxes are very unstable.

Then, I keep trying test the second best, the third best parameters. I finally found the parameter that good for this.

```python
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 10  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 4 # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
```

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?



![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?



![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my test video result](./project_video.mp4)

Here's a [link to my video result](./project_video.mp4)

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

There are several challenge in this project. First is how to find out the best parameters combination for question. I used a script to run all parameters and find the best one for this question. It tooks about 12 hours to finish all combinations in my desktop computer. 

The second challenge is choosing a good sliding windows size and offset from far to near distance for every frame.
So I decide to use 1x size of windows on farthest then plus 0.5x for every 2 steps. So the scales will be from 1 to 3.5 in my code. The reason I choice for different size with different distance is perspective of lane. The size of car near is bigger than far one. Farthermore, I also limit the sliding window range from 400 to 760 in y direction. This is the highly probability range that cars will appear.

The final challenge is getting stable or smoothing bounding boxes for cars. I try to use previous 10 frames, and cumulate heat map of 10 frames, and I also increase threshold to 6. That means it should have at least 6 frame contains car detection, and it will be considered as real true for car detection.

For this project, the performace is also a big issue. It tooks almost 18 mins to generate a 50 second video. For this reason, it's very hard to use my code in real-world application. To implement the whole pipeline with other faster language(C/C++) will be important for this.
