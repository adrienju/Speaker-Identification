from collections import deque
import cv2
import time
import glob
import random
import tkinter as tk
from tkinter import *
from tkinter import messagebox
from tkinter import filedialog
import shutil
from tkinter import *
from threading import Thread
from pydub import AudioSegment
from matplotlib import pyplot as plot
from PIL import Image, ImageDraw
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import PIL.Image
import pyglet

filetest = ''

def quit_(root):
    root.destroy()
def plotwav(datapath):
    
    audio = AudioSegment.from_file(datapath)
    data = np.fromstring(audio._data, np.int16)
    fs = audio.frame_rate
    BARS = 100
    BAR_HEIGHT = 60
    LINE_WIDTH = 5
    length = len(data)
    RATIO = length/BARS
    count = 0
    maximum_item = 0
    max_array = []
    highest_line = 0
    for d in data:
        if count < RATIO:
            count = count + 1
            if abs(d) > maximum_item:
                maximum_item = abs(d)
        else:
            max_array.append(maximum_item)
            if maximum_item > highest_line:
                highest_line = maximum_item
            maximum_item = 0
            count = 1
    line_ratio = highest_line/BAR_HEIGHT
    im = Image.new('RGBA', (BARS * LINE_WIDTH, BAR_HEIGHT), (255, 255, 255, 1))
    draw = ImageDraw.Draw(im)
    current_x = 1
    for item in max_array:
        item_height = item/line_ratio
        current_y = (BAR_HEIGHT - item_height)/2
        draw.line((current_x, current_y, current_x, current_y + item_height), fill=(169, 171, 172), width=4)
        current_x = current_x + LINE_WIDTH
    im.save('TASOEUR.png'
                )

def on_closing():
    if messagebox.askokcancel("Quit", "Do you want to quit?"):
        root.destroy()
 
def startPlaying(datadir):
    music_file = pyglet.media.load(datadir)
    player.queue(music_file)
    player.play()
    pyglet.app.run()

def playSound(datadir):
    global sound_thread 
    sound_thread = Thread(target=startPlaying(datadir))
    sound_thread.start()

def generate_data():
    copydata(filetest)
    os.system("python /home/adrienj/Desktop/AudioNet-master/scripts/make_testdata.py")
    print("Test data generated ")

def set_filename():
    filename = filedialog.askopenfilename(initialdir = "/home/adrienj/Desktop")
    print(filename)
    global filetest 
    filetest = filename
    return filename
    
def copydata(filetest):
    print(filetest)
    shutil.copy(filetest, '/home/adrienj/Desktop/AudioNet-master/scripts/tf_files/data_audio/filetotest')
    plotwav(filetest)

def create_image():
    root.photo2 = ImageTk.PhotoImage(Image.open('TASOEUR.png'))
    vlabel.configure(image= root.photo2)

def getlistimg():
    l = []
    folders = []
    folders=glob.glob('/home/adrienj/Desktop/AudioNet-master/scripts/tf_files/data_audio/filetotest/*.jpg')
    if len(folders) == 2:
        folders.append(folders[1])
        l = random.sample(folders, 3)
    if len(folders) == 1:       
        folders.append(folders[0])
        folders.append(folders[0])
        l = random.sample(folders, 3)
    if len(folders) >= 3:
        l = random.sample(folders, 3)
    return(l)

def clean():
    os.system('python cleantest')

def plotmfcc():
    l = getlistimg()
    root.mfcc1 = ImageTk.PhotoImage(Image.open(l[0]))
    root.mfcc2 = ImageTk.PhotoImage(Image.open(l[1]))
    root.mfcc3 = ImageTk.PhotoImage(Image.open(l[2]))
    labelmfcc1.configure(image= root.mfcc1)
    labelmfcc2.configure(image= root.mfcc2)
    labelmfcc3.configure(image= root.mfcc3)


if __name__ == '__main__':
    getlistimg()
    root = tk.Tk()
    root.title('Voice')
    player = pyglet.media.Player()
    

    w = 1500 # width for the Tk root
    h = 900 # height for the Tk root

    # get screen width and height
    ws = root.winfo_screenwidth() # width of the screen
    hs = root.winfo_screenheight() # height of the screen

    # calculate x and y coordinates for the Tk root window
    x = (ws/2) - (w/2)
    y = (hs/2) - (h/2)


    # set the dimensions of the screen
    # and where it is placed
    root.geometry('%dx%d+%d+%d' % (w, h, x, y))
    root.grid_rowconfigure(1, weight=1)
    root.grid_columnconfigure(0, weight=1)
    
 

    
    
    #Graph
    root.photowav = ImageTk.PhotoImage(Image.open('blank.png'))
    vlabel = tk.Label(root, image=root.photowav)
    vlabel.pack()

    root.mfcc1 = ImageTk.PhotoImage(Image.open('blankmfcc.png'))
    labelmfcc1 = tk.Label(root, image=root.mfcc1)
    labelmfcc1.pack()
    
    root.mfcc2 = ImageTk.PhotoImage(Image.open('blankmfcc.png'))
    labelmfcc2 = tk.Label(root, image=root.mfcc2)
    labelmfcc2.pack()
    
    root.mfcc3 = ImageTk.PhotoImage(Image.open('blankmfcc.png'))
    labelmfcc3 = tk.Label(root, image=root.mfcc3)
    labelmfcc3.pack()
   
    Frame1 = Frame(root, width=1400, height=80, bg='blue')
    Frame1.pack(fill=None, expand=False, side='top')
    Label(Frame1, text="Speaker Identification using CNN").pack(padx=10, pady=10)

    Logo = Frame(root, width=100, height=80, bg='orange')
    Logo.pack(fill=None, expand=False, side='right')

    Framefileselector = Frame(root, width=100, height=80, bg='orange')
    Framefileselector.pack(fill=None, expand=False, side='left')

    Frame2 = Frame(root, width=20, height=20, bg='red', padx=100, pady=100)
    Frame2.place(bordermode=OUTSIDE, height=100, width=100)
    Frame2.pack(fill=None, expand=False, side='bottom')


    button1 = Button(root, text="Play the mp3 file", command=playSound)
    button1.pack()

    button2 = Button(root, text="Generer les images", command=generate_data)
    button2.pack()

    button2 = Button(root, text="Afficher test data infos", command=create_image)
    button2.pack()

    buttonmfcc = Button(root, text="Afficher les MFCCS", command=plotmfcc)
    buttonmfcc.pack()
    
    buttonclean = Button(root, text="Start a new test", command=clean)
    buttonclean.pack()


    button = Button(Framefileselector, text='Choose a mp3 file to use', command=set_filename)
    button.pack()
    print(button) #filapath
    #print(fileName)


   
 
    # setup the update callback
    root.protocol("WM_DELETE_WINDOW",on_closing)
    root.mainloop()
