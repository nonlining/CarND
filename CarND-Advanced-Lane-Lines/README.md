# Advanced Lane Finding Project

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./chess1.png "image1"
[image2]: ./undistorted_1.png "image2"
[image3]: ./undistorted_2.png "image3"
[image4]: ./apply_g_v2.png "image4"
[image5]: ./binary_example.png "image5"
[image6]: ./p_transform.png "image6"
[image7]: ./examples/color_fit_lines.jpg "image7"
[video1]: ./project4_adv_lane_video.mp4 "Video"

---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in "project4.ipynb".  

I used 2 OpenCV functions, findChessboardCorners and calibrateCamera to implement Camera Calibration. First, using findChessboardCorners to find the internal chessboard corners with parameter x_corners = 9 and y_corners = 6. Then, the return value of findChessboardCorners fed into calibrateCamera to calibrate camera.

The following images are different angles of Chess board image with highlighed corners.

![alt text][image1]

These are applying calibrateCamera function on Chess board image results.

![alt text][image2]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

I apply the distortion correction to example images:

![alt text][image3]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I try several ways and did many experiments to get the better binary image, include sharpen image, get channel from HLS and LAB. But I did not used gradients technique on my images. This is because gradient technique did not help much to get clear binary images.

My color transforms function is named, **apply_color_gradient_v2( )**. This function get channel L from HLS color space and channel B for LAB color space. This combination were came from a lot experiments. The following is my result.

![alt text][image4]
![alt text][image5]

The whole color and transform transform pipeline are as following python code.
```python

def processing_pipeline(img, mtx, dist, vertices, src, dst):
    
    gblur = cv2.GaussianBlur(img, (5,5), 20.0)
    img = cv2.addWeighted(img, 2, gblur, -1, 0)
    
    img = region_of_interest(img, vertices)
    
    img = undistort_img(img, mtx, dist)    
    
    img, M, Minv = perspective_transform(img, src, dst)
    
    img = apply_color_gradient_v2(img)
    
    return img, Minv

```

I applied sharpen image first, then crop the image region of road. Sharpen image can help get more clear cut binary image, and remove unwanted region can help faster image processing. I did teh perspective transform before color transform. This order is by my experiments that can get better image.

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code of perspective transform is on my project4.ipynb ** 
Apply a perspective transform to rectify binary image ("birds-eye view") ** block. The code contain the following source (src) and destination (dst) points. These 2 points will be used in getPerspectiveTransform() function.

```python

src = np.float32([(575,464),
                  (707,464), 
                  (258,682), 
                  (1049,682)])

dst = np.float32([(450,0),
                  (w-450,0),
                  (450,h),
                  (w-450,h)])
                  
```
This is the result.

![alt text][image6]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

This is the math for fitting lane lines with a 2nd order polynomial:

![alt text][image7]




#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

There are 2 parts of this project I took most of time to tune. One is Finding a good color transforms, gradients and other methods to create a thresholded binary image. And the other is finding a method to verify invalid lane information (radius of curvature ,and the position of lane) to abandon.





