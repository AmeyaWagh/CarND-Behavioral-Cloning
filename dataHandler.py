import csv
import os
from model import *
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import cv2
import numpy as np

DATASET_PATH = "/home/ameya/mydata/behavioral_cloning_data"
CSV_PATH = os.path.join(DATASET_PATH,"driving_log.csv")


data = []
with open(CSV_PATH,'rb') as fp:
	reader = csv.reader(fp)
	for line in reader:
		# print(line)
		data.append(line)


shuffle(data)

num_samples = len(data)

print(len(data[0]))
for i,sample in enumerate(data[0]):
	print(i,sample)

X_data = []
Y_data = []
for sample in data:
	center_img = cv2.imread(sample[0])
	left_img = cv2.imread(sample[1])
	right_img = cv2.imread(sample[2])
	steering = float(sample[3])
	X_data.append(center_img)
	X_data.append(left_img)
	X_data.append(right_img)
	Y_data.append([steering])
	Y_data.append([steering+0.2])
	Y_data.append([steering-0.2])

print(len(X_data),len(Y_data))
X_data = np.array(X_data)
Y_data = np.array(Y_data)

X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.2)


car_model = get_model()

car_model.fit(x=X_train, y=y_train, batch_size=20, validation_split=0.2, verbose=1, nb_epoch=5, shuffle=True)
score  = car_model.evaluate(x=X_test, y=y_test, batch_size=20, verbose=1)
print(car_model.metrics_names)
print(score)
car_model.save('model.h5')


if __name__ == '__main__':
	print(CSV_PATH)
	# print(data[0])