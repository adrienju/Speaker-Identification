import sys
import webrtcvad
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read, write

def apply_webrtc_vad(signal, sample_rate, frame_duration, agressiveness=3):
    vad = webrtcvad.Vad(agressiveness)
        
    frame_size = np.int(sample_rate * frame_duration / 1000)
    nb_frames = np.int(len(signal) / frame_size)
    signal_clean_vad = []
    no_speech = []
    for i in range(0, nb_frames):
        if vad.is_speech(signal[i*frame_size:(i+1)*frame_size], sample_rate) is True:
            signal_clean_vad = np.append(signal_clean_vad, signal[i*frame_size:(i+1)*frame_size-1])
            no_speech = np.append(no_speech, np.zeros(frame_size))
        else:
            no_speech = np.append(no_speech, np.ones(frame_size))
    signal_clean_vad = np.divide(signal_clean_vad, max(signal_clean_vad))
    max_signal = max(signal)
    for i in range(0, len(no_speech)):
        no_speech[i] = no_speech[i]*max_signal
    return signal_clean_vad, no_speech


if __name__ == "__main__":
    save = False
    figure = False
    if '-s' in sys.argv or '--save' in sys.argv:
        save = True
    if '-f' in sys.argv or '--figure' in sys.argv:
        figure = True
    if '-h' in sys.argv or '--help' in sys.argv or len(sys.argv) < 2:
        print("Silence removal\nThis script can be used to remove silence")
        print("-h / --help: to display this message")
        print("-f / --figure: to display and save a comparison figure")
        print("-s / --save: to save the signal without silence")
    else:    
        for option in sys.argv:
            if option.endswith('.wav'):
                print(option)
                [sample_rate, signal] = read(option)
                filename = option.replace('.wav', '_vad')
                try:
                    signal_vad, no_speech = apply_webrtc_vad(signal, sample_rate, 20)
                    if figure is True:
                        f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(18, 12), dpi= 80, facecolor='w', edgecolor='k')
                        ax1.plot(signal)
                        ax1.plot(no_speech)
                        ax1.set_title("Signal original")
                        ax1.set_xlabel('Échantillon')
                        ax1.set_ylabel('Amplitude')
                        ax2.plot(signal_vad)
                        ax2.set_title("Signal sans silence")
                        ax2.set_xlabel('Échantillon')
                        ax2.set_ylabel('Amplitude')
                        plt.savefig(filename + '.png', bbox_inches='tight')
                    if save is True:
                        write(filename + '.wav', sample_rate, signal_vad)        
                except: 
                    print("The file doesn't match the constraint of WebRTC.")
