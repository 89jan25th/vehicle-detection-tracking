**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car-hog1.png
[image2]: ./output_images/car-hog2.png
[image3]: ./output_images/car-hog3.png
[image4]: ./output_images/notcar-hog1.png
[image5]: ./output_images/notcar-hog2.png
[image6]: ./output_images/notcar-hog3.png
[image7]: ./output_images/car_hist_1.JPG
[image8]: ./output_images/car_hist_2.JPG
[image9]: ./output_images/not_car_hist_1.JPG
[image10]: ./output_images/not_car_hist_2.JPG
[image11]: ./output_images/test1_map.png
[image12]: ./output_images/test2_map.png
[image13]: ./output_images/test3_map.png
[image14]: ./output_images/test4_map.png
[image15]: ./output_images/test5_map.png
[image16]: ./output_images/test6_map.png
[image17]: ./output_images/test1_map_label.png
[image18]: ./output_images/test2_map_label.png
[image19]: ./output_images/test3_map_label.png
[image20]: ./output_images/test4_map_label.png
[image21]: ./output_images/test5_map_label.png
[image22]: ./output_images/test6_map_label.png


[video1]: https://github.com/89jan25th/vehicle-detection-tracking/blob/master/project_video_proc_thre2.mp4?raw=true
[video1]: https://github.com/89jan25th/vehicle-detection-tracking/blob/master/project_video_proc_thre3.mp4?raw=true


### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The corresponding code is under '2. HOG feature extraction and visualization' in pipeline.ipynb.
I extracted HOG features by using the method  from the course that used 'skimage.feature.'  
I used graysacle image for HOG feature extraction and the parameters are set as below.

- orient = 9  # HOG orientations  
- pix_per_cell = 8 # HOG pixels per cell  
- cell_per_block = 2 # HOG cells per block  
- hog_channel = 0 # Can be 0, 1, 2, or "ALL"  

Below is the examples of cars and not cars' HOG features with the parameters above.

These parameters are set and used for both training and image processing.  
  
HOG example | HOG example
:-------------------------:|:-------------------------:
car example 1         |  car example 2
![alt text][image1] |  ![alt text][image2]
car example 3         |  not car example 1
![alt text][image3] | ![alt text][image4]
not car example 2         |  not car example 3
![alt text][image5] | ![alt text][image6]


#### 2. Explain how you settled on your final choice of HOG parameters.

I set my parameters based on the course, and I finetuned them a bit to see if there is significant difference at result, but there wasn't.  
Just for the note, I had to use only one hog channel because of my laptop's low specification. :(


#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The code consists of three parts '3. Spatial and color feature extraction, and visualization', '4. Feature extraction function,' and '5. Training.'

  
1) I used three features for training.  
  
- spatial feature: 16 * 16 image is used to lower the calculation task.  
- HOG feature: already described.  
- color histogram feature:  
  First I compared RGB and YCrCb histogram charts between car and not car images.
  I found, in the YCrCb space, Cr and Cb has a pretty similar pattern in both car and not car images but Y is quite outstanding.
  Thereofre I chose YCrCb for color histogram feature extraction.  
  
Color histogram | Color histogram
:-------------------------:|:-------------------------:
car example 1         |  not-car example 1
![alt text][image7] |  ![alt text][image9]
car example 2         |  not-car example 2
![alt text][image8] | ![alt text][image10]
  
2) For training, SVM is used. The kinds of library I used for training are as below.

>from sklearn.preprocessing import StandardScaler  
>from sklearn.svm import LinearSVC  
>from sklearn.model_selection import train_test_split  

Feature vector length is 2628, and training takes about 34 seconds in my system, and the test accuracy of SVC is 0.9885. 


### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The corresponding code is under '6. Sliding window and car detection.'  
The sliding window function is also from the course.

1) Here's the responsible block of codes.

> nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1  
> nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1  
> nfeat_per_block = orient \* cell_per_block \*\* 2  
    
> \# 64 was the orginal sampling rate, with 8 cells and 8 pix per cell  
> window = 64  
> nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1  
> cells_per_step = 2  \# Instead of overlap, define how many cells to step  
> nxsteps = (nxblocks - nblocks_per_window) // cells_per_step  
> nysteps = (nyblocks - nblocks_per_window) // cells_per_step  

> for xb in range(nxsteps):  
>>    for yb in range(nysteps):  
>>>       ypos = yb*cells_per_step  
>>>       xpos = xb*cells_per_step  
>>>       .... HOG feature extraction and prediction goes on

2) scale  
I learned that from 1 to 2.5 the scale is useful from trials. 1 is most useful for a small(or far) object but also can contribute to find a large (in a middle distance). But as the scale gets closer to 2.5, it is almost only useful for a large(near) object.  
Therefore I used four different scales (1, 1.5, 2, 2.5) and combine them when video processing.

3) overlap  
by setting cells_per_step to 2, the overlap ratio is 75%. This number was set by the instructor and I didn't really change it seems just reasonable.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

My pipeline starts with predicting and bounding.

Here are six frames and their corresponding heatmaps:

column1  | column2 
:-------------------------:|:-------------------------:
test1        |  test2
![alt text][image11] |  ![alt text][image12]
test3        |  test4
![alt text][image13] | ![alt text][image14]
test5        |  test6
![alt text][image15] | ![alt text][image16]


Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
  
column1   | column2   
:-------------------------:|:-------------------------:
test1        |  test2
![alt text][image17] |  ![alt text][image18]
test3        |  test4
![alt text][image19] | ![alt text][image20]
test5        |  test6
![alt text][image21] | ![alt text][image22]


---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [result video with thresold 2](./project_video_thre2.mp4) and  [with thresold 3](./project_video_thre3.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

Two method are used for preventing false postives:  
1) Thresholding,  
2) Averaging.
Both is under '7. Image processing (heatmap and thresholding.'

1) Thresholding:  
Thresholding is simply to set a threshold on the heat values. Too high threshold causes false negatives(where car isn't recognized) and too low threshold causes false postives(where not car is recognized).  
With threshold 2, there are several frames with false postives, and with 3, there are several frames with false negatives.

2) Averaging:  

>if n_count%2 == 0:  
>    n_count += 1  
>    box_list = find_box(img)  
>    heat_list.append(find_heat(img, box_list))  
>    if len(heat_list) > 20:  
>        heat_list.pop(0)  
>    heat_sum = np.zeros_like(heat_list[0])  
>    for pair in heat_list:  
>        heat_sum += pair  
>     heat_sum = heat_sum // len(heat_list) # calculate the average  

here's the explanation.  
- n_count%2 is for skipping every other frame.  
- then every box appends in a list  
- when it exceeds 20 then oldest one is removed  
- and calculate the average.  

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

My averaging method is calculated for an individual object but all the bounding boxes. This problem can be solved by introducing class and track vehicles. My implementation doesn't include 'tracking' feature.  
My code also easily failes when the car is passing the scene with guardrail under shades. This can be solved by using these false postives 
as training data.
