from keras.models import Sequential
from keras.layers import Flatten , Dense , Lambda , Activation, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.models import load_model
from keras.optimizers import Adam

def get_model(learning_rate=1e-4):
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1.0 ,input_shape= (160,320,3)))
    
    # model.add(Cropping2D(cropping = ((50,20) ,(0,0))))
    model.add(Convolution2D(24,5,5, subsample = (2,2),activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(1, 1)))

    model.add(Convolution2D(36,5,5, subsample = (2,2),activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(1, 1)))

    model.add(Convolution2D(46,5,5,subsample = (2,2) ,activation ='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(1, 1)))

    model.add(Convolution2D(64,3,3, activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(1, 1)))

    model.add(Convolution2D(64,3,3,activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(1, 1)))

    model.add(Flatten())
    # model.add(Dropout(0.5))
    model.add(Dense(1164))
    model.add(Activation('relu'))
    
    model.add(Dense(100))
    # model.add(Dropout(0.3))
    model.add(Activation('relu'))

    model.add(Dense(50))
    # model.add(Dropout(0.2))
    model.add(Activation('relu'))

    model.add(Dense(10))
    model.add(Activation('relu'))

    model.add(Dense(1))
    model.compile(loss = 'mse', optimizer = Adam(learning_rate))
    return model