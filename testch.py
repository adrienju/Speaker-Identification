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

def makechunks(path):
    folders=glob.glob(path+'*')
    
    for folder in folders:
        waves = glob.glob(folder+'/'+ '*.wav') #liste des fichiers .wav
        print ('w',waves)
        if len(waves) == 0:
            return 10
        for i in waves: # pour tous les fichiers .wav
            print("i =", i)
            w = i
            print("w =", w)
            myaudio = AudioSegment.from_file(i, 'wav')
            chunk_length_ms = 300
            chunks = make_chunks(myaudio, chunk_length_ms)
            print (chunks)
            for i, chunk in enumerate(chunks):
                chunk_name = w.split('.')[0] + "chunk{0}.wav".format(i)
                print (chunk_name)
                print ("exporting", chunk_name)
                chunk.export(folder+'/'+chunk_name, format="wav")

if __name__ == '__main__':
    path='./tf_files/tf_files/data_audio/'
    parser = argparse.ArgumentParser()
    #parser.add_argument('path', help="Specify the path to the music directory", default="../tf_files/data_mp3/")
    parser.add_argument('--mkchunks', help="Set this flag if you want to make chunks of waves", action="store_true", default=True)
    args = parser.parse_args()
    if args.mkchunks:
        print(
        "Searching for wav files in :", path)
        try:
            r = makechunks(path)
            if r == 10:
                print(
                "No wav files in given path")
            else:
                print(
                "Completed successfully")
        except Exception as e:
            print(
            "Something went wrong : ", e)
    