# **Behavioral Cloning** 

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

---

<div style="text-align:center"><img src=./assets/final_run.gif width="600" height="400"></div>

---
#### Youtube Links

<div style="text-align:center;">
	<span style="display: inline-block;">
	<a href=https://www.youtube.com/watch?v=XiGJvkqjmzU&feature=youtu.be>
		<img src=https://img.youtube.com/vi/XiGJvkqjmzU/0.jpg width="300" height="200">
	</a>
	<a href=https://www.youtube.com/watch?v=nuFGrUctFHo&feature=youtu.be>
		<img src=https://img.youtube.com/vi/nuFGrUctFHo/0.jpg width="300" height="200">
	</a>
	</span>
</div>

---

### Files Submitted & Code Quality

<!-- #### 1. Submission includes all required files and can be used to run the simulator in autonomous mode -->

My project includes the following files:
```
.
├── assets
├── dataHandler.py
├── drive.py
├── model.h5
├── model.py
├── README.md
├── train.py
└── video.py

```

* `model.py` - containing the [Nvidia End to End learning](https://arxiv.org/pdf/1604.07316.pdf) model implemented in Keras. 
* `drive.py` for driving the car in autonomous mode
* `model.h5` containing a trained convolution neural network 
* `dataHandler.py` contains generators to generate and augment training and validation data
* `train.py` contains the training script which utilizes `model.py` and 'dataHandler.py' to train the model

<!-- #### 2. Submission includes functional code -->
To run the model.
Start the simulator in Autonomous mode and in a different terminal execute the following command 
```sh
python drive.py model.h5 imgs/
```


The [Nvidia End to End learning](https://arxiv.org/pdf/1604.07316.pdf) model in `model.py` is defined in Keras and is given by the function:

```python
def get_model(learning_rate=1e-4):
    model = Sequential()

    model.add(Lambda(lambda x: x/127.5 - 1.0 ,input_shape= (160,320,3)))
    
    model.add(Cropping2D(cropping = ((65,25) ,(0,0))))

    model.add(Convolution2D(24,5,5, subsample = (2,2),activation = 'relu'))

    model.add(Convolution2D(36,5,5, subsample = (2,2),activation = 'relu'))

    model.add(Convolution2D(48,5,5,subsample = (2,2) ,activation ='relu'))

    model.add(Convolution2D(64,3,3, subsample=(1, 1), activation = 'relu'))

    model.add(Convolution2D(64,3,3, subsample=(1, 1), activation='relu'))

    model.add(Flatten())
    
    model.add(Dense(1164))
    model.add(Dropout(0.1))
    model.add(Activation('relu'))
    
    model.add(Dense(100))
    model.add(Dropout(0.1))
    model.add(Activation('relu'))

    model.add(Dense(50))
    model.add(Dropout(0.1))
    model.add(Activation('relu'))

    model.add(Dense(10))
    model.add(Activation('relu'))

    model.add(Dense(1))

    model.summary()
    
    model.compile(loss = 'mse', optimizer = Adam(learning_rate))
    return model
```
<!-- #### 3. Submission code is usable and readable -->

<!-- The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works. -->



### Model Architecture and Training Strategy

The following is the model architecture summary


```
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
lambda_1 (Lambda)                (None, 160, 320, 3)   0           lambda_input_1[0][0]             
____________________________________________________________________________________________________
cropping2d_1 (Cropping2D)        (None, 70, 320, 3)    0           lambda_1[0][0]                   
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 33, 158, 24)   1824        cropping2d_1[0][0]               
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 15, 77, 36)    21636       convolution2d_1[0][0]            
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 6, 37, 48)     43248       convolution2d_2[0][0]            
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 4, 35, 64)     27712       convolution2d_3[0][0]            
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 2, 33, 64)     36928       convolution2d_4[0][0]            
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 4224)          0           convolution2d_5[0][0]            
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 1164)          4917900     flatten_1[0][0]                  
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 1164)          0           dense_1[0][0]                    
____________________________________________________________________________________________________
activation_1 (Activation)        (None, 1164)          0           dropout_1[0][0]                  
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 100)           116500      activation_1[0][0]               
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 100)           0           dense_2[0][0]                    
____________________________________________________________________________________________________
activation_2 (Activation)        (None, 100)           0           dropout_2[0][0]                  
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 50)            5050        activation_2[0][0]               
____________________________________________________________________________________________________
dropout_3 (Dropout)              (None, 50)            0           dense_3[0][0]                    
____________________________________________________________________________________________________
activation_3 (Activation)        (None, 50)            0           dropout_3[0][0]                  
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 10)            510         activation_3[0][0]               
____________________________________________________________________________________________________
activation_4 (Activation)        (None, 10)            0           dense_4[0][0]                    
____________________________________________________________________________________________________
dense_5 (Dense)                  (None, 1)             11          activation_4[0][0]               
====================================================================================================
Total params: 5,171,319
Trainable params: 5,171,319
Non-trainable params: 0
____________________________________________________________________________________________________

```



<!-- #### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 18-24) 

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18). 


#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.


#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).


#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section.  -->

As it can be seen in the summary, the model has `5,171,319` which are trainable, has 5 convolution layers and 5 fully connected layer. The image is cropped in the pipeline itself using the `Cropping2D` layer in Keras.

#### Dataset generation and Data Augmentation

The [ Udacity self-driving-car-sim ](https://github.com/udacity/self-driving-car-sim) was used in Training mode to capture images of while driving. There are 3 cameras, `Center`, `Left` and `Right` positioned accordingly on the car. The steering angle is recorded simultaneously and are time synchronized and saved in `driving_log.csv`

```sh
.
├── driving_log.csv
└── IMG

```



Here is an example of log from the `driving_log.csv`

| Center Image | Left Image | Right Image | steering | throttle | break | speed |
|:-------------|:-----------|:------------|:---------|:---------|:------|:------|
| /home/ameya/mydata/behavioral_cloning_data/IMG/center_2018_05_23_19_36_50_138.jpg |	/home/ameya/mydata/behavioral_cloning_data/IMG/left_2018_05_23_19_36_50_138.jpg	| /home/ameya/mydata/behavioral_cloning_data/IMG/right_2018_05_23_19_36_50_138.jpg	| 0	| 0	| 0	| 4.301273E-06 |


| Left Image | Center Image | Right Image |
|:----------:|:------------:|:-----------:|
| <img src=./assets/left.jpg width="300" height="200"> | <img src=./assets/center.jpg width="300" height="200"> | <img src=./assets/right.jpg width="300" height="200"> |



The row in the `driving_log.csv` define the no of samples collected by driving in the driving simulator. The car was driven few laps in the simulator keeping it centered in the track and few more in the opposite direction to make the model less bias to a side, in this case left. To add some adversaries,  some samples were collected by aggressive driving and also driving in the other track which had sharp turns. The dataset was the divided into `Training` and `Testing` by splitting it into 80% and 20% respectively. The actaul samples in training and testing set are given below

```
('train_samples', 32553)
('test_samples', 8139)
```

The `dataHandler.py` uses the `driving_log.csv` to load respective images. For each sample 3 images are loaded, namely Left, right and center. while the simulator in autonomous mode only uses the center camera. Thus a steering correction of `0.245` is compansated in the LEFT and RIGHT images. these images are also flipped and angles are negated to augment the data. Thus for every sample in  `driving_log.csv`, we obtain 5 data samples each containing an image and it's corresponding angle.
```python
def flip_image(image,angle):
    return np.fliplr(image), -1*angle
```

```python
def augment_image(sample,X_data,Y_data):
    center_img = cv2.imread(sample[0])
    left_img = cv2.imread(sample[1])
    right_img = cv2.imread(sample[2])
    steering = float(sample[3])

    X_data.append(center_img)
    Y_data.append([steering])
    
    X_data.append(left_img)
    Y_data.append([steering+STEERING_CONST])
    
    X_data.append(right_img)
    Y_data.append([steering-STEERING_CONST])

    flip_left, left_angle = flip_image(left_img, steering + STEERING_CONST)

    X_data.append(flip_left)
    Y_data.append([left_angle])

    flip_right, right_angle = flip_image(right_img, steering - STEERING_CONST)

    X_data.append(flip_right)
    Y_data.append([right_angle])
```


To reduce memory while loading dataset for training, generators are used.

#### Training
Training the model is handled by `train.py` . The training dataset and testing dataset were shuffled thorougly and the testing dataset was used to validate the model while training.

```
SAMPLES_PER_EPOCH = 30000
EPOCHS = 8
VALIDATION_SAMPLES = 6400
LEARNING_RATE = 1e-4
```

The parameters were obtained by trial and error. It is generally observed that the effective learning rate for ADAM optimizer is 1e-4. It was also observed that After 8 epochs, the validation loss increased . This was an indication of overfitting.The mean square error loss obtained was approximately `0.036` and the validation loss was `0.032`. The model was then saved to `model.h5` to be used for inference. 


|	 |	Plot of the training history |
|:----:|:-----:|
| training loss | <div style="text-align:center;"><img src=./assets/training_loss.png width="300" height="200"></div> |
|				| no of epochs |

#### Testing
The model was tested by running the simulator in the autonomous mode and using `model.h5` for inference. Initially the model was just trained for 2 laps and 2 epochs to see if the model was working correctly. Later more data was collected by driving aggressively and in the opposite direction. This helped the model to generalize and barely make some sharp turns. about 2 laps of data was collected from the other track which had very sharp turns. this made the model better in turns of taking smooth turns and avoiding crashes.



---


<!-- ### Model Architecture and Training Strategy

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

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary. -->



