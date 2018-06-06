import numpy as np
from PIL import Image,ImageTk
#from interface import display_pics
import tkinter as tk
global WIDTH
global HEIGHT
global NB
import cv2

WIDTH = 512
HEIGHT = 480
NB = 3

class photobook:
# Class containing images taken by user
    def __init__(self,nb):
        self.count = 0
        self.im = np.zeros(shape=(HEIGHT,WIDTH,3,nb),dtype=np.uint8)
        self.var = tk.IntVar(value=1)
        self.choice = 0
        self.param = {'style': 0,'couleurs': 0}
        self.cam = cv2.VideoCapture(0)
    def add_im(self,im):
        self.im[:,:,:,self.count] = im
        self.count = self.count + 1
        if self.count == NB:
            self.count = 0
    def view_im(self):
        for i in range(0,3):
            img = Image.fromarray(self.im[:,:,:,i],'RGB')
            img.show()

