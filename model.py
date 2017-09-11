import csv
import cv2
import numpy as np
import glob

T1_C = 3
T1_CC = 2
T1_R = 0
T2 = 1

def read_images(dir_name, flip=False):
    lines = []
    with open(dir_name + '/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

    print('Number of images for ' + dir_name + ': ' + str(len(lines)))
    if flip:
        print('Flipping images and measurements is on')

    images = []
    measurements = []
    for line in lines:
        source_path = line[0]
        file_name = source_path.split('/')[-1]
        current_path = dir_name + '/IMG/' + file_name
        image = cv2.imread(current_path)
        images.append(image)
        measurement = float(line[3])
        measurements.append(measurement)

        if flip:
            images.append(np.fliplr(image))
            measurements.append(-measurement)

    return (images, measurements)

def create_lenet(model):
    model.add(Convolution2D(6, 5, 5, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(6, 5, 5, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dropout(0.5))
    model.add(Dense(84))
    model.add(Dropout(0.5))
    model.add(Dense(1))

def create_nvidia_net(model, dropout=False):
    model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    if dropout:
        model.add(Dropout(0.5))
    model.add(Dense(50))
    if dropout:
        model.add(Dropout(0.5))
    model.add(Dense(10))
    if dropout:
        model.add(Dropout(0.5))
    model.add(Dense(1))

nb_epochs = 4
do_dropout = True

dirs = glob.glob('../data/*')

final_images = []
final_measurements = []

# Images for track 1 - going clockwise
res = read_images(dirs[T1_C], flip=False)
final_images.extend(res[0])
final_measurements.extend(res[1])

# Images for track 1 - going counter clockwise
res = read_images(dirs[T1_CC], flip=False)
final_images.extend(res[0])
final_measurements.extend(res[1])

# Images for track 1 - recovery
res = read_images(dirs[T1_R], flip=True)
final_images.extend(res[0])
final_measurements.extend(res[1])

# Images for track 2
res = read_images(dirs[T2], flip=False)
final_images.extend(res[0])
final_measurements.extend(res[1])

print('Total number of images: ' + str(len(final_images)))

X_train = np.array(final_images)
y_train = np.array(final_measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))

# LeNet
# create_lenet(model)

# NVIDIA Net
create_nvidia_net(model, dropout=do_dropout)

model.compile(loss='mse', optimizer='adam')

#import matplotlib.pyplot as plt

history = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=nb_epochs)

model.save('model.h5')

### plot the training and validation loss for each epoch
#plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
#plt.title('model mean squared error loss')
#plt.ylabel('mean squared error loss')
#plt.xlabel('epoch')
#plt.legend(['training set', 'validation set'], loc='upper right')
#plt.show()
