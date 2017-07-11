# **Traffic Sign Recognition**

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"

[image4]: ./web-signs/1*Iiwrp2CLW7bhOopz7QTB5w.png "Stop sign"
[image6]: ./web-signs/220px-stop_sign.jpg "stop sign"
[image7]: ./web-signs/3918169fbec64aa0eba9fb1d212fc128.jpg "General caution"
[image8]: ./web-signs/mifuUb0.jpg "Road work"
[image5]: ./web-signs/100_1607.jpg "Right of way"
[image9]: ./web-signs/slippery-road-sign-440.jpg "Slippery road"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/ciju/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Mostly just rendered a few images. Didn't try any exploratory
statistics.

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

Decided to normalize image because it might help with better backprop works better with it. 

From what I understand, normalization reduces oscillation in the and helps with better backprop convergence. 

Used `tf.image.per_image_standardization` in the pipeline. Seems to
have been sufficient from normalization perspective. No other
augmentation was done. 

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model, is mostly based on LeNet, with the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x3 RGB image   							|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x16 				|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x32   				|
| Fully connected		| 800 -> 240   									|
| RELU					|												|
| Dropout				| Keep probability of 0.5						|
| Fully connected		| 240 -> 84   									|
| RELU					|												|
| Dropout				| Keep probability of 0.75						|
| Fully connected		| 84 -> 43   									|
| Softmax				| etc.        									|
|						|												|


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

EPOCHS = 14
BATCH_SIZE = 128

Initialization parameters:
    mu = 0
    sigma = 0.1

Loss function multiplier = 0.0001
Learning rate = 0.001

Used `AdamOptimizer`.

Mostly trial and error.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 96%
* test set accuracy of 95%


Q: Why was this particular model chosen?
A: Choose LeNet, mostly because that was the architecture we were introducted to. And because its built around (introduced?) convolutions, which are good at 'understanding' images.

Q: Why did you think that this model would be an appropriate choice for this project?
A: First, its a classification problem. Second, image problem. Third, the number of different properties is about the order of magnitude of what LeNet was designed for. Seemed like LeNet should be able to get the few extra percentage precisions needed (with dropout and input regularization). And it did.

Q: What specific modifications were made from the original architecture? (If any)
A: Two most important changes where to normalize input data (using `tf.image.per_image_standardization`) and to use dropout with variable keep probabilities. .5 after first fully connected layer, and .75 after second, to help with regularization. 

Q: What were the results of any tests that you performed?
A: I had tried a few test. Dropouts in initial convolution layers. Chaning the learning rate. Chaning EPOCHs. Most of it didn't help with accuricy much. Etc. I guess thats the important part. I didn't have a systematic way to do changes and keep track of their effect. Most of the effort seemed to be in forming a theory, testing it out and then forming another etc.

If an iterative approach was chosen:
Most of it was about trying different things. Probably the two most
important decisions were normalization and dropout. Tried a changing a
few parameters and increasing the width of the layers etc. Didn't get
significantly better results.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are six German traffic signs that I found on the web:

![alt text][image4]
I would have guessed that this stop sign would have been difficult to predict (because of the background being different, and because the model wasn't trained with any kind of screws or other image preprocessing). But model predicted it with high accuracy.

![alt text][image6]
This one was predicted wrong. It is a surprise. My guess is, the dark sides somehow make model believe its Priority road. 

![alt text][image7]
![alt text][image8]
![alt text][image5]
These 3 images model shouldn't have much problems with. 

![alt text][image9]
This image is slightly clipped at top and quite close filling the whole image. I would have guessed the model shouldn't have trouble predicting it, but it predicted it wrong. And wasn't sure about the prediction.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Stop Sign      		| Stop sign   									|
| Stop Sign      		| Priority road									|
| General caution     	| general caution 										|
| Road work				| Road work											|
| Right of way	  		| Right of way  			 				|
| Slippery Road			| Pedestrian      							|

Accuracy on these data is low, compared to the test data. The images do go through the same preprocessing. The issue seems to be that there is not enough preprocessing in terms of transforms on the image. Ex, rotate the images a little. Crop out/in a little. I would guess that it would do better with more data (both by preprocessing and more real data)

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .99         			| Stop sign   									|
| .99         			| Stop sign   									|
| .99     				| general caution 										|
| .97					| Road work											|
| .99	      			| Right of way					 				|
| .54				    | Slippery Road      							|

Seems like the model was quite certain about most predictions. Even the wrong ones. Example the wrong Stop sign prediction. Only in case of Slippery road the prediction was not certain. I guess, the model needs more work. Ex, stop sign seems to be classified as priority road. Seems to me, that it has picked sharp cuts on the side as priority road.
