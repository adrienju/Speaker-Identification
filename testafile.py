from __future__ import print_function
import keras
import glob
import wave
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.utils import plot_model
from keras import backend as K
from sklearn.model_selection import train_test_split
from PIL import Image
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import skimage
from sklearn import preprocessing
import skimage.io as io
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import wavfile

import csv
img_rows, img_cols = 130, 131
batch_size = 16
num_classes = 5
np.set_printoptions(threshold=np.inf, suppress=True)
#K.set_image_dim_ordering('th')
#########################################
def readcsv(csvtest):
    reader = csv.reader(open(csvtest), delimiter=',')
    data = []
    iternum=0
    listeimg = []
    listlabel = []
    listpath = []
    #print(reader)
    for row in reader:
        data.append(row)
        listeimg.append(data[iternum][0])
        listlabel.append(data[iternum][1])
        listpath.append(data[iternum][0])
        iternum += 1
        
     #
    vec = np.array([np.array(Image.open(fname)) for fname in listeimg])
    print("VEC =",vec.shape)  #return label and vec of all images
    label = np.asarray(listlabel)
  
    return vec, label, listpath

def loadimages():
    X, y, listpath = readcsv('/home/adrienj/Desktop/AudioNet-master/scripts/dataset_test.csv')
    onehot = pd.get_dummies(y)
    y_voice = onehot.as_matrix()
    return X, y, listpath

def loadparameters(filepath):
    filepath = open('/home/adrienj/Desktop/AudioNet-master/scripts/varmean.txt','r')
    f = filepath.readlines()
    xmean = f[0]
    xvar = f[1]
    xmean = xmean.strip('\n')
    xvar = xvar.strip('\n')
    xmeanxvar = xmean+' '+xvar
    return xmeanxvar 

def sortlist(name, liste): #liste pour un user
    newlist = []
    for elements in liste: 
        if name == elements.split('/')[8]:
            newlist.append(elements)
    return newlist


def testaguy(classe, biglist, xmeanxvar, dic, nbclasses): # Prend en entrée LA classe a tester, la liste CSV , les param pour normalisation, le dic et le nb de classes
    newlist = sortlist(classe, biglist)
    model = load_a_model('/home/adrienj/Desktop/AudioNet-master/scripts/voicekerasmodel.h5')
    loss = 0 
    predict = 0
    for image in newlist:
        vectoryhat = testing(image, model, xmeanxvar)  
        vectory, currentclass = classvector(image, dic, nbclasses) #return un np array type 1 hot
        loss = computeloss(vectory, vectoryhat) #compare les deux vecteurs
        if loss < 0.50:
            print(image)
        predict += loss
    predict = predict/len(newlist)
    print("For the class: "+classe, predict)

def testwhole(dic, biglist, xmeanxvar, nbclasses):
    for elements in dic:
        testaguy(elements, biglist, xmeanxvar,dic ,nbclasses)
        

def testing(imagepath,model,xmeanxvar): #generate yhat vector
    allval = np.fromstring(xmeanxvar, sep=' ')
    xmean = allval[0].astype('float32')
    xvar = allval[1].astype('float32')
    image = np.array([np.array(Image.open(imagepath))])
    image = image.astype('float32')
    image = (image-xmean)/xvar
    classes = model.predict_proba(image, batch_size=1)
    return classes
    

def classvector(path , dic, nbofclasses): #return y 
    name = path.split('/')[3]
    valindex = dic.get(name)
    result = np.zeros((nbofclasses,1))
    result[valindex] = 1
    return result, valindex 

def computeloss(vectory, vectoryhat): #compute losse entre 2 vec
    result = np.dot(vectory.T, vectoryhat.T)
    return result 

def testafile(biglist, xmeanxvar, nbofclasses):
    a = np.zeros((1,nbofclasses))
    print(nbofclasses)
    newlist2 = sortlist('filetotest', biglist)
    model = load_a_model('/home/adrienj/Desktop/AudioNet-master/scripts/voicekerasmodel.h5')
    for image in newlist2:
        a = np.concatenate((a, testing(image, model, xmeanxvar)))
    a = np.sum(a, axis=0)
    a /= len(newlist2)
    print(a)
    return a
      
def returnresult(vec, numberofclasses, dic):
    a = np.arange(numberofclasses)
    plt.bar(a, vec, alpha=0.8)
    plt.xticks(a, dic.keys(), rotation='45')
    print(dic.keys())
    plt.ylabel("Probability")
    plt.title("Histogram of estimated probability")
    plt.savefig('/home/adrienj/Desktop/AudioNet-master/scripts/app/static/images/histo.png')
    plt.show()

def plotwav(filename):
    samplerate, data = wavfile.read(filename)
    times = np.arange(len(data))/float(samplerate)

    # Make the plot
    # You can tweak the figsize (width, height) in inches
    plt.figure(figsize=(30, 4))
    plt.fill_between(times, data[:,0], data[:,1], color='k') 
    plt.xlim(times[0], times[-1])
    plt.xlabel('time (s)')
    plt.ylabel('amplitude')
    # You can set the format by changing the extension
    # like .pdf, .svg, .eps
    plt.savefig('./static/images/wavselected.png')
    plt.show()



def get_list_class():
    path = '/home/adrienj/Desktop/AudioNet-master/scripts/tf_files/tf_files/data_audio/'
    dic = {}
    folders=sorted(glob.glob(path+'*'))
    nbofclasses = len(folders)
    value = 0 
    for folder in folders:
        test = folder.split('/')[-1]
        if test != 'filetotest':
            dic[test] = value
            value +=1 
    print(dic)
    return dic, nbofclasses


# load the model we saved
def load_a_model(modeltoload):
    model = load_model(modeltoload)
    return model

# predicting multiple images at once
if __name__ == '__main__':
    w = 0
    X, y, datapath = loadimages()
    xmeanxvar = loadparameters('/home/adrienj/Desktop/AudioNet-master/scripts/varmean.txt')
    dic, nbofclasses = get_list_class()
   
    returnresult(testafile(datapath, xmeanxvar, nbofclasses), nbofclasses, dic)
    #testaguy('pauline', datapath, xmeanxvar, dic, nbofclasses)
    #testwhole(dic, datapath, xmeanxvar, nbofclasses)
    

