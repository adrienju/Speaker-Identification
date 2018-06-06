'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import plot_model
from keras import backend as K
from sklearn.model_selection import train_test_split
import numpy as np
from PIL import Image
import skimage
from sklearn import preprocessing
import skimage.io as io
import matplotlib.pyplot as plt
import pandas as pd
import csv
img_rows, img_cols = 256, 254
batch_size = 32
num_classes = 4
epochs = 300
np.set_printoptions(threshold=np.inf)
#K.set_image_dim_ordering('th')
#########################################
def readcsv(csvtest):
    reader = csv.reader(open(csvtest), delimiter=',')
    data = []
    iternum=0
    listeimg = []
    listlabel = []
    #print(reader)
    for row in reader:
        data.append(row)
        listeimg.append(data[iternum][0])
        listlabel.append(data[iternum][1])
        iternum += 1
     
     #   print(data[iternum][0]) 
    vec = np.array([np.array(Image.open(fname)) for fname in listeimg])
    print("VEC =",vec.shape)
    label = np.asarray(listlabel)
  
    return vec, label

#def splitdataset(vec, labeltrans):

print("READING CSV ....")
X, y = readcsv('dataset.csv')
print("iNFO ABOUT DATASET ....")
print("X shape", X.shape)
print("HOT ENCODING .............")
onehot = pd.get_dummies(y)
y_voice = onehot.as_matrix()
print("SPLITTING DATASET............")
x_train_voice, x_test_voice, y_train_voice, y_test_voice = train_test_split(X, y_voice, test_size=0.2)
print("Training on ", x_train_voice.shape)
print("Testing on ", x_test_voice.shape)

print("DATA NORMALAZATION............")
x_train_voice = x_train_voice.astype('float32')
x_test_voice = x_test_voice.astype('float32')
x_train_voice /= 255
x_test_voice  /= 255

#x_train_voice = x_train_voice.reshape(x_train_voice.shape[0], img_rows, img_cols,3 ,1)
#x_test_voice = x_test_voice.reshape(x_test_voice.shape[0], img_rows, img_cols,3,1)
input_shape = (img_cols, img_rows, 3)
print("Xtrain voice = ", x_train_voice.shape)
#########################################
# input image dimensions



"""
num_classes2 = 10
# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
y_train = keras.utils.to_categorical(y_train, num_classes2)
y_test = keras.utils.to_categorical(y_test, num_classes2)

plt.imshow(x_train[9])
plt.show()
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
"""
model = Sequential()
model.add(Conv2D(32, kernel_size=(4, 4),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (4, 4), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(25, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

plot_model(model, show_shapes='True', show_layer_names='True')

history = model.fit(x_train_voice, y_train_voice,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_split=0.3)
score = model.evaluate(x_test_voice, y_test_voice, verbose=0)

print(y_test_voice)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()