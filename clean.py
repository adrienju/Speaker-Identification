import os
import glob
import shutil
import subprocess
import argparse
from pydub import AudioSegment
from pydub.utils import make_chunks
from scipy.io import wavfile
from matplotlib import pyplot as plt
import sys
from PIL import Image
sys.modules['Image'] = Image
path='./tf_files/tf_files/data_audio/'
folders=glob.glob(path+'**/', recursive='True')
waves = []
for folder in folders:
    waves += glob.glob(folder+'/'+ '*.jpg')
    waves += glob.glob(folder+'/'+ '*wav*')
    waves += glob.glob(folder+'/'+ '*.mp3')
    print ('w',waves)
for elements in waves:
    os.remove(elements)
 

    
    