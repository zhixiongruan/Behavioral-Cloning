## Behavioral Cloning Writeup ##

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image2]: ./examples/center.jpg "Center Image"
[image3]: ./examples/left.jpg "Left Image"
[image4]: ./examples/right.jpg "Right Image"
[image6]: ./examples/before.jpg "Normal Image"
[image7]: ./examples/after.jpg "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 32 and 64, which is the same model as NVIDIA. (model.py lines 75-88)

The model includes RELU layers to introduce nonlinearity (model.py lines 78-82), and the data is normalized in the model using a Keras lambda layer (model.py line 76), also, Dropout layer (model.py line 83)

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 83).

The model was trained and validated on different data sets to ensure that the model was not overfitting (model.py line 90-91). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 90).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. Also, I re-used the each image flipped as additional data.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to end-to-end artchitecture by [NVIDIA](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/). I thought this model might be appropriate because of its better result for the accuracy.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model by adding Dropout so that it could mitigate the overfitting situation.  

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I added the training data for these spots.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 75-88) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture :

|Layer  | Description | Parameters |
|-------|-------------|------------|
|Layer 1| Lambda| normalization, input=160x320x3 | 
|Layer 2| Cropping2D| crop rows, top=70, bottom=25, new size=65x320x3 | 
|Layer 3| CNN 24x5x5 | relu activation|
|Layer 4| CNN 36x5x5 | relu activation|
|Layer 5| CNN 48x5x5 | relu activation|
|Layer 6| CNN 64x3x3 | relu activation|
|Layer 7| CNN 64x3x3 | relu activation|
|Layer 8| Dropout |0.2|
|Layer 9| Flatten ||
|Layer 10| Dense |100|
|Layer 11| Dense |50|
|Layer 12| Dense |10|
|Layer 13| Dense |1|

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to the situation it need to modify its angle. These images show what a recovery looks like starting from on the line both side of the road :

![alt text][image3]
![alt text][image4]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would mitigate the diviation of the data, because there are almost left curves only in the first track. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]


After the collection process, I had 50,250 number of data points. I then preprocessed this data by Gaussian Filter.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by the matured accuracy level. I used an adam optimizer so that manually training the learning rate wasn't necessary.
