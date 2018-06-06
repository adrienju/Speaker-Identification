import os
import glob
import shutil
import subprocess
import librosa
import librosa.display
import csv
import argparse
import wave
from pydub import AudioSegment
from pydub.utils import make_chunks
from scipy.io import wavfile
from matplotlib import pyplot as plt
import numpy as np
import sys
from PIL import Image
sys.modules['Image'] = Image

def get_duration(wav):
    f = wave.open(wav, 'r')
    frames = f.getnframes()
    rate = f.getframerate()
    duration = frames / float(rate)
    f.close()
    return duration


def generatecsv(listelem):
    f = open('/home/adrienj/Desktop/AudioNet-master/scripts/dataset_test.csv', 'w')
    with f:
        writer = csv.writer(f)
        for row in listelem:
            writer.writerow(row)

def mp3towav(path):
    folders=glob.glob(path+'*')
    print ("folders",folders)
    for folder in folders:
      files = glob.glob(folder+'/**/'+ '*.mp3', recursive=True)
      if len(files) == 0:
          return 10
      for file in files:
          mp = file
          wa = file.replace('mp3', 'wav')
          subprocess.call(['sox', mp, '-e', 'mu-law', '-r', '16k', wa, 'remix', '1'])

def makechunks(path):
    count = 0
    chunk_lim = 550
    folders=glob.glob(path+'*')
    for folder in folders:
      waves = glob.glob(folder+'/**/*.wav', recursive=True)
      print ('w',waves)
      if len(waves) == 0:
          print("makechunks vide")
          return 10
      for i in waves:
          count += 1 
          w = i
          myaudio = AudioSegment.from_file(i)
          chunk_length_ms = 3000
          chunks = make_chunks(myaudio, chunk_length_ms)
          for i, chunk in enumerate(chunks):
              chunk_name = str(count)+"chunk{0}.wav".format(i)
              if len(chunk) > chunk_lim:
                print("FOLDER =",folder)
                print("chunck name =",chunk_name)
                chunk.export(folder+'/'+chunk_name, format="wav")
               


def graph_spectrogram(wav_file):
    rate, data = get_wav_info(wav_file)
    y, sr = librosa.load(wav_file)
    librosa.feature.melspectrogram(y=y, sr=sr)
    D = np.abs(librosa.stft(y))**2
    S = librosa.feature.melspectrogram(S=D)

    # Passing through arguments to the Mel filters
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,fmax=8000)
    plt.figure(figsize=(1, 1.30), frameon=False)
    plt.axis('off')
    librosa.display.specshow(librosa.power_to_db(S,ref=np.max), fmax=8000,hop_length=160)
    plt.tight_layout(pad=0)
    plt.savefig(wav_file.split('.wav')[0] + '.jpg', # Dots per inch
                dpi =100,
                frameon=False
                )  # Spectrogram saved as a .png
    #if os.path.exists(wav_file.split('.wav')[0] + '.png'):
    #    os.remove(wav_file.split('.wav')[0] + '.png')
    plt.close()

def get_wav_info(wav_file):
    rate, data = wavfile.read(wav_file)
    return rate, data

def get_dataset_info(path):
    waves = glob.glob('/home/adrienj/Desktop/AudioNet-master/scripts/tf_files/data_audio/**/*chunk*.wav', recursive=True)
    return len(waves)


def wav2spectrogram(path):
    listelem = [[] for i in range (get_dataset_info(path))]
    countlist = 0
    folders = glob.glob(path+'*')
    num = 0
    for folder in folders:
        print("FOR THE CLASS Named", folders[num].split('/')[-1])
        waves = glob.glob(folder+'/' + '*chunk*.wav')
        print(waves)
        for f in waves:
            listelem[countlist].extend((f.replace("wav","jpg"),folders[num].split('/')[-1]))
            countlist += 1
            try:
                print("Generating spectrograms..")
                graph_spectrogram(f)
            except Exception as e:
                print("Something went wrong while generating spectrogram: ", e)
        num += 1
    print(listelem)
    return listelem



if __name__ == '__main__':
    path='/home/adrienj/Desktop/AudioNet-master/scripts/tf_files/data_audio/'
    parser = argparse.ArgumentParser()
    #parser.add_argument('path', help="Specify the path to the music directory", default="../tf_files/data_mp3/")
    parser.add_argument('--mkchunks', help="Set this flag if you want to make chunks of waves", action="store_true", default=True)
    parser.add_argument('--mp3towav', help="Set this flag if you want to convert mp3 to wav", action="store_true",default=True)
    parser.add_argument('--spectrogram', help="Set this flags  to create spectrograms", action="store_true",default=True)
    parser.add_argument('--clean', help="Remoove all the blank", action="store_true",default=False)
    args = parser.parse_args()
    if args.mp3towav:
        print(
        "Path : ", path)
        try:
            r = mp3towav(path)
            if r == 10:
                print
                "No mp3i files in specified directory"
            else: 
                print(
            "Searching for wav files in :", path)
                print(
                "All mp3 files processed completely")
        except Exception as e:
            print(
            "Something went wrong :", e)
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
    if args.spectrogram:
        print("Finding files in : ", path)
        r = wav2spectrogram(path)
        generatecsv(r)
