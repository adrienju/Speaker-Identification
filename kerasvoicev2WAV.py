'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.utils import plot_model
from keras import backend as K
from sklearn.model_selection import train_test_split

import os
from PIL import Image
import skimage
from sklearn import preprocessing
import skimage.io as io
import matplotlib.pyplot as plt
import pandas as pd
import csv
img_rows, img_cols = 130, 131
batch_size = 32
limit = 100
num_classes = len([i for i in os.listdir("/home/adrienj/Desktop/AudioNet-master/scripts/tf_files/tf_files/data_audio")])
dim = int((num_classes*10)/2)
epochs = 25

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


def testanimage(fname, model, xmean, xvar):
  
    image = np.array([np.array(Image.open(fname))])
    image = image.astype('float32')
    image = (image-xmean)/xvar
    classes = model.predict_classes(image, batch_size=1)
    print(classes)



def datainfo(a):
    info = np.sum(a, axis=0)
    print(info)
    print(np.sum(info, axis=0))
    print(np.sum(info, axis=0)/12)
    w=0
    for i in info:
        print("In the class "+str(w), info[w])
        w += 1

def saveparameters(xmean, xvar):
    file = open('varmean.txt','w') 
    file.write(str(xmean)+'\n')
    file.write(str(xvar)+'\n')
    file.close()

#def splitdataset(vec, labeltrans):
print("NUMBER OF CLASSES =", num_classes)
print("READING CSV ....")
X, y = readcsv('dataset.csv')
print("iNFO ABOUT DATASET ....")
print("X shape", X.shape)
print("HOT ENCODING .............")
onehot = pd.get_dummies(y)
y_voice = onehot.as_matrix()
print("SPLITTING DATASET............")
x_train_voice, x_test_voice, y_train_voice, y_test_voice = train_test_split(X, y_voice, test_size=0.1, random_state=3)
#x_train_voice, x_test_voice, y_train_voice, y_test_voice = train_test_split(X, y_voice, test_size=0.19, random_state=True)
print("Training on ", x_train_voice.shape)
print("Testing on ", x_test_voice.shape)

print("DATA NORMALAZATION............")
print(x_train_voice.shape)
x_train_voice = x_train_voice.astype('float32')
x_test_voice = x_test_voice.astype('float32')
xmean = np.mean(x_train_voice)
xvar = np.std(x_train_voice)
x_train_voice = (x_train_voice-xmean)/xvar
x_test_voice = (x_test_voice-xmean)/xvar
print("Xtrain voice = ", x_train_voice.shape)

#x_train_voice = x_train_voice[0:500]
#y_train_voice = y_train_voice[0:500]
#x_train_voice, x_osef = train_test_split(x_train_voice, y_train_voice, test_size=0.5, random_state=3)

print("Xtrain voice after slicing= ", x_train_voice.shape)
#########################################



model = Sequential()
model.add(ZeroPadding2D((1,1),input_shape=(130,100,3)))
model.add(Convolution2D(32, 3, 3, activation='relu'))
model.add(MaxPooling2D((4,4), strides=(4,4)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(MaxPooling2D((4,4), strides=(4,4)))


model.add(Flatten())
model.add(Dense(10*num_classes, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(dim, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))


model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              #optimizer=keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True),
              metrics=['accuracy'])

plot_model(model, show_shapes='True', show_layer_names='True')

history = model.fit(x_train_voice, y_train_voice,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_split=0.10,
          shuffle=False)
score = model.evaluate(x_test_voice, y_test_voice, verbose=1)
print("FOR THE TRAINING")
datainfo(y_train_voice)
print("FOR THE TEST")
datainfo(y_test_voice)

model.save("voicekerasmodel.h5")
print(y_test_voice.shape)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
saveparameters(xmean, xvar)
print(type(xmean))


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