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
path='/home/adrienj/Desktop/AudioNet-master/scripts/tf_files/data_audio/filetotest'
folderss=glob.glob(path+'/*')
print(folderss)
for folder in folderss:
    os.remove(folder)
 

    
    