import os
import csv
import pandas as pd
import numpy as np
import cv2
import matplotlib.pylab as plt
np.random.seed(902)


# Read in the data csv in to a panda dataframe
df = pd.read_csv("./data/driving_log.csv")
print("Number of frames from each camera:", len(df))

# Balancing Data
# Create a new DataFrame after Downsampling the images that with low steering angle
steer_straight_idx = df[(df.steering >= -0.045) & (df.steering <= 0.045)].index
downsample_idx = np.random.choice(steer_straight_idx,int(len(steer_straight_idx)*.05), )
new_idx = np.concatenate([df[~((df.steering >= -0.045) & (df.steering <= 0.045))].index, downsample_idx])
new_df = df.iloc[np.sort(new_idx)].reset_index(drop=True)

# Make list of image-steering angle pair
# Perform steering angle correction on left & right mounted camera frame
correction=0.2
samples = []
samples = np.vstack([np.hstack([new_df.left.values, 
                                new_df.center.values, 
                                new_df.right.values]),
                     np.hstack([new_df.steering+correction, 
                                new_df.steering, 
                                new_df.steering-correction])]).T.tolist()

# ### Data Augmentation
from skimage import transform #scikit-image for image transform API

# #### 1. Randomize Lightning Conditions
def modify_sat_lum(image, uni_low=0.4, uni_high=1.6):
    sat_f, lum_f =np.random.uniform(uni_low, uni_high, 2).tolist()
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS).astype(float)
    hls[:,:,1] = hls[:,:,1] * lum_f
    hls[:,:,1][hls[:,:,1] > 255] = 255
    hls[:,:,2] = hls[:,:,2] * sat_f
    hls[:,:,2][hls[:,:,2] > 255] = 255
    return cv2.cvtColor(hls.astype('uint8'),cv2.COLOR_HLS2BGR)

# #### 2. Cast Shadow
def cast_shadow(image):
    # randomly select shadow side
    shadow_side = np.random.choice(['left','right','top', 'bottom'])

    if (shadow_side == 'top') | (shadow_side == 'bottom'):
        divider_left, divider_right = np.random.randint(0, image.shape[0], 2)
    if (shadow_side == 'left') | (shadow_side == 'right'):
        divider_top, divider_bot = np.random.randint(0, image.shape[1], 2)
        
    if shadow_side == 'top':
        ul = [0, 0]
        ur = [image.shape[1], 0]
        ll = [0, divider_left]
        lr = [image.shape[1], divider_right]
    elif shadow_side == 'bottom':
        ul = [0, divider_left]
        ur = [image.shape[1], divider_right]
        ll = [0, image.shape[0]]
        lr = [image.shape[1], image.shape[0]]
    elif shadow_side == 'left':
        ul = [0, 0]
        ur = [divider_top, 0]
        ll = [0, image.shape[0]]
        lr = [divider_bot, image.shape[0]]
    elif shadow_side == 'right':
        ul = [divider_top, 0]
        ur = [image.shape[1], 0]
        ll = [divider_bot, image.shape[0]]
        lr = [image.shape[1], image.shape[0]]

    # form shadow vertices
    vertices = [np.array([ll,ul,ur,lr])]

    # create a blank mask
    mask = np.zeros_like(image[:,:,0])   

    #filling pixels inside the polygon defined by "vertices" with the fill color    
    mask = cv2.fillPoly(mask, vertices, 255).astype('bool')

    # shadow-masked area will be halved its luminanse
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    hls[:,:,1][mask]=hls[:,:,1][mask] * np.random.uniform(0.1, 0.3, 1)

    return cv2.cvtColor(hls.astype('uint8'),cv2.COLOR_HLS2BGR)

# #### 3. Shift Horinzontal and Vertical
def horizontal_vertical_shift(image, steering, hlim=[-60,60], vlim=[-20,20]):
    hlim=sorted(hlim); vlim=sorted(vlim)
    hv_trans= [np.random.randint(hlim[0], hlim[1], 1)[0],
               np.random.randint(vlim[0], vlim[1], 1)[0]]
    trans_img = transform.warp(image, transform.AffineTransform(translation=hv_trans))
    steering = steering + hv_trans[0]*-0.007
    return (trans_img*255).astype('uint8'), steering

# #### 4. Rotation
# Reference: https://stackoverflow.com/questions/25895587/python-skimage-transform-affinetransform-rotation-center

def rotate(image, steering, deglim=[-15,15], center=[160,70]):
    degree = np.random.randint(deglim[0], deglim[1], 1)[0]
    # shift the center of single point perspective to origin
    shift_x,shift_y = center
    tf_shift = transform.SimilarityTransform(translation=[-shift_x, -shift_y])

    # perform rotation
    tf_rotate = transform.SimilarityTransform(rotation=np.deg2rad(degree))

    # shifting the center of single point spective to its original location
    tf_shift_inv = transform.SimilarityTransform(translation=[shift_x, shift_y])

    # Warp into final image
    rot_img = transform.warp(image, (tf_shift + (tf_rotate + tf_shift_inv)).inverse)

    _steering = steering + degree*0.02
    return (rot_img*255).astype('uint8'), _steering

# ### Generators
import sklearn
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size = 0.2)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: #loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            
            images = []
            measurements = []
            
            for batch_sample in batch_samples:
                # Make the file path for center, left and right images
                imgpth = './data/IMG/' + batch_sample[0].split('/')[-1]
                img = cv2.imread(imgpth)
                steering = float(batch_sample[1])
                
                # Chances of perform image augmentation
                if np.random.binomial(1, 0.7) > 0:
                    if np.random.choice([True, False]):
                        if np.random.choice([True, False]):
                            # Random rotation
                            img, steering = rotate(img, steering)
                        else:
                            # Random translation
                            img, steering = horizontal_vertical_shift(img, steering)
                    
                    # Randon lighting conditions
                    if np.random.choice([True, False]):
                        img = modify_sat_lum(img)

                    # Shadow Casting
                    if np.random.choice([True, False]):
                        img = cast_shadow(img)                        
                
                images.append(img)
                measurements.append(steering)

            X_train = np.array(images)
            y_train = np.array(measurements)
            yield sklearn.utils.shuffle(X_train, y_train)


# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

# ### Model Training in Keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPool2D
from keras.callbacks import ModelCheckpoint

model = Sequential()

# Crop Image
x = model.add(Cropping2D(cropping=((70,20), (0,0)), input_shape=(160,320,3)))

# pixel value normalization
model.add(Lambda(lambda x: (x/255.0) - 0.5))

# Conv Layers
model.add(Conv2D(24, (5, 5), strides=(2,2), padding='same', activation='relu'))
model.add(Dropout(0.1, seed=902))
model.add(Conv2D(36, (5, 5), strides=(2,2), padding='same', activation='relu'))
model.add(Dropout(0.1, seed=902))
model.add(Conv2D(48, (5, 5), strides=(2,2), padding='same', activation='relu'))
model.add(Dropout(0.1, seed=902))
model.add(Conv2D(64, (3, 3), strides=(3,3), padding='same', activation='relu'))
model.add(Dropout(0.1, seed=902))
model.add(Conv2D(64, (3, 3), strides=(3,3), padding='same', activation='relu'))
model.add(Dropout(0.1, seed=902))
model.add(Flatten())

# FC Layers
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
model.compile(loss='mse', optimizer = 'adam')# a FC network

filepath="trained_models/commaai-newaug-eph{epoch:02d}-val{val_loss:.3f}.h5"
checkpoint = ModelCheckpoint(filepath)
callbacks_list = [checkpoint]

history_object = model.fit_generator(train_generator, 
                                     steps_per_epoch=len(train_samples)*2/32, 
                                     validation_data=validation_generator, 
                                     validation_steps=len(validation_samples)*2/32, 
                                     epochs=10, 
                                     callbacks=callbacks_list)
print(model.summary())

