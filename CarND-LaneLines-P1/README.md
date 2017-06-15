# **Finding Lane Lines on the Road** 

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./test_images_output/solidWhiteCurve.jpg "Result"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

There are 5 steps in my pipeline
 1. Transfer RGB image to Grayscale image.
 2. Apply Gaussian Blur on Grayscale image.
 3. Apply Canny transform.
 4. Mask the region that outside the current line.
 5. Use Hugh Line and draw_line() function to draw lines.

I modify draw_line() in following way:
 1. average all right lane slopes to get a mean slope.
 2. average all right lane x and y points that I can have a point can be used to calculate intercept of a line.
 3. Use this line to draw a line from the bottom of imgae to far point.

If you'd like to include images to show how the pipeline works, here is how to include an image: 

![alt text][image1]


### 2. Identify potential shortcomings with your current pipeline

I tested my code on Optional Challenge video, the result is not good. 2 lines intersect at far point, and the slopes of 2 lines are not correct.

### 3. Suggest possible improvements to your pipeline

