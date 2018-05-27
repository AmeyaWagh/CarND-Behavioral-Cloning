import csv
import os
from model import *
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import cv2
import numpy as np

DATASET_PATH = "/home/ameya/mydata/behavioral_cloning_data"
CSV_PATH = os.path.join(DATASET_PATH,"driving_log.csv")
BATCH_SIZE = 64

CENTER_IMAGE = 0
LEFT_IMAGE = 1
RIGHT_IMAGE = 2
STEERING = 3
STEERING_CONST = 0.2

def load_files():
    data = []
    with open(CSV_PATH,'rb') as fp:
        reader = csv.reader(fp)
        for line in reader:
            data.append(line)

    data = shuffle(data)
    data = shuffle(data)
    data = shuffle(data)
    train_data, test_data = train_test_split(data, test_size = 0.2 )
    train_samples = len(train_data)
    test_samples = len(test_data)
    return train_data, train_samples,test_data,test_samples

def flip_image(image,angle):
    return np.fliplr(image), -1*angle


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


train_data, train_samples,test_data,test_samples = load_files()

def fetch_data(data_type='train'):
    while True:
        if data_type == 'train':
            data, num_samples = train_data,train_samples
        else:
            data, num_samples = test_data,test_samples

        shuffle(data)
        for i in xrange(0,num_samples,BATCH_SIZE):
            X_data = []
            Y_data = []
            for sample in data[i:i+BATCH_SIZE]:
                augment_image(sample, X_data, Y_data)

            print(len(X_data),len(Y_data))
            X_data = np.array(X_data)
            Y_data = np.array(Y_data)

            yield shuffle(X_data,Y_data)


if __name__ == '__main__':
    print(CSV_PATH)
    # print(data[0])