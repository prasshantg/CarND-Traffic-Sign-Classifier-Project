#**Traffic Sign Recognition** 

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./training_data.png
[image2]: ./validation_data.png
[image3]: ./test_data.png
[image4]: ./grayscale_image.png
[image5]: ./original_image.png
[image6]: ./rotated_image.png
[image7]: ./Test_image_0.png
[image8]: ./Test_image_1.png
[image9]: ./Test_image_2.png
[image10]: ./Test_image_3.png
[image11]: ./Test_image_4.png
[image12]: ./probability_image_0.png
[image13]: ./probability_image_1.png
[image14]: ./probability_image_2.png
[image15]: ./probability_image_3.png
[image16]: ./probability_image_4.png


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

You're reading it! and here is a link to my [project code](https://github.com/prasshantg/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

The code for this step is contained in the second code cell of the IPython notebook.  

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

The code for this step is contained in the third code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed

![Training data][image1]
![Validation data][image2]
![Testing data][image3]

###Design and Test a Model Architecture

The code for image preprocessing is contained in the fourth code cell of the IPython notebook.

As a first step, I decided to convert the images to grayscale because it reduces number of parameters to process. Also, 

Here is an example of a traffic sign image before and after grayscaling.

![Grayscale image][image4]

The forth code cell of the IPython notebook contains the code for augmenting the data set. I decided to generate additional data because some images had very less number of samples such as pedestrian. To add more data to the the data set, I skimage transform pakcage to rotate images by different angles. I created new images only for images with low number of samples. 

Here is an example of an original image and an augmented image:

![Original Image][image5]
![Rotated Image][image6]

The code for my final model is located in the seventh cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   					| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x36 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x36 				|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x36 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x36 					|
| Convolution 1x1     	| 1x1 stride, valid padding, outputs 5x5x16 	|
| RELU					|												|
| Fully connected		| outputs 1x1x120								|
| RELU					|												|
| Dropout				|												|
| Fully connected		| outputs 1x1x96								|
| RELU					|												|
| Dropout				|												|
| Fully connected		| outputs 1x1x96								|
| RELU					|												|
| Fully connected		| outputs 1x1x43								|
| Softmax				|												|
|						|												|
|						|												|
 

The code for training the model is located in the eigth cell of the ipython notebook. 

To train the model, I used below parameters

Epochs = 40
Batch size = 128
AdamOptimizer
Learning rate = 0.001
Dropout probability = 0.5 for training and 1.0 for evaluating


The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

I started with base as LeNet. Added dropout layers after FC layers as the dataset is sparse. I chose LeNet as it was simple and to start and also looked ideal for small images with not many features.

I tried increasing number of channels in convolution operations to big number but that caused poor accuracy. Mostly because more number of parameters than features. Instead of wide layers I then tried with deep layers by adding one more FC and convolution layer with less number of channels. It gave me good accuracy for both training and validation dataset.

Also, I tried changing learning rate for each epoch alternatively to high and low learning rate. Did not find any improvement in accuracy but learning time increased with it. Hence dropped at the end.

My final model results were:
* training set accuracy of 0.995
* validation set accuracy of 0.967
* test set accuracy of 0.927

Here are five German traffic signs that I found on the web:

![Yield][image7] ![No passing][image8] ![Speed limit (70km/h)][image9] 
![No entry][image10] ![Speed limit (120km/h)][image11]

The code for making predictions on my final model is located in the 14th cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Yield      			| Yield   										| 
| No passing     		| No passing									|
| 70 km/h				| 70 km/h										|
| No entry				| No entry						 				|
| 120 km/h				| 120 km/h		      							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of ...

The code for making predictions on my final model is located in the 14th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .54         			| Yield   										| 
| .26     				| Go straight or right 							|
| .17					| No vehicles									|
| .11	      			| Bumpy road					 				|
| .10				    | Keep right      								|

![Probability graph][image12]

For the second image ... 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .31         			| No passing   									| 
| .14     				| No passing for vehicles over 3.5 metric tons 	|
| .12					| Vehicles over 3.5 metric tons prohibited		|
| .09	      			| End of no passing					 			|
| .07				    | Dangerous curve to the left      				|

![Probability graph][image13]

For the third image ... 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .18         			| Speed limit (70km/h)   						| 
| .06     				| Speed limit (20km/h) 							|
| .05					| Stop											|
| .05	      			| No vehicles					 				|
| .05				    | Speed limit (30km/h)      					|

![Probability graph][image14]

For the forth image ... 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .22         			| No entry   									| 
| .06     				| Go straight or right 							|
| .06					| Go straight or left							|
| .06	      			| Turn left ahead				 				|
| .04				    | End of no passing      						|

![Probability graph][image15]

For the firth image ... 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .21         			| Speed limit (120km/h)   						| 
| .07     				| Speed limit (100km/h) 						|
| .07					| Speed limit (20km/h)							|
| .07	      			| Speed limit (70km/h)			 				|
| .07				    | Go straight or left      						|

![Probability graph][image16]

