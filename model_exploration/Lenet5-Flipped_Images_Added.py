
# coding: utf-8

# In[7]:

import csv
import cv2
import numpy as np

lines = []

with open('../data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []

for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = '../data/IMG/' + filename
    image = cv2.imread(current_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)

augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(measurement)
    
X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)


# In[ ]:

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPool2D


# In[ ]:

model = Sequential()
model.add(Lambda(lambda x: (x/255.0) - 0.5, input_shape=(160,320,3)))
model.add(Conv2D(6, (5, 5), activation='relu'))
model.add(MaxPool2D())
model.add(Conv2D(32, (5, 5), activation='relu'))
model.add(MaxPool2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))
model.compile(loss='mse', optimizer = 'adam')# a FC network
from keras.models import Sequential
from keras.layers import Flatten, Dense
model.fit(X_train,y_train, validation_split=0.2, shuffle=True, initial_epoch=0, epochs=5)

model.save('Lenet5-Flipped_images.h5')
print(model.summary())


# In[ ]:



