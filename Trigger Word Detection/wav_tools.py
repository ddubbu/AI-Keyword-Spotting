import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
import numpy as np
import os
from pydub import AudioSegment  # for manipulating audio data.

# Calculate and plot spectrogram for a wav audio file
def graph_spectrogram(wav_file):
    rate, data = get_wav_info(wav_file)
    nfft = 200 # Length of each window segment
    fs = 8000 # Sampling frequencies
    noverlap = 120 # Overlap between windows
    nchannels = data.ndim
    if nchannels == 1:
        pxx, freqs, bins, im = plt.specgram(data, nfft, fs, noverlap = noverlap)
    elif nchannels == 2:
        pxx, freqs, bins, im = plt.specgram(data[:,0], nfft, fs, noverlap = noverlap)

    plt.ylabel("how active each frequency")
    plt.xlabel("number of time-steps")
    plt.title(wav_file)
    ''' plot 띄우면 정지되어서 일시적으로 주석 '''
    # plt.show()  # see spectogram

    return pxx

# Load a wav file
def get_wav_info(wav_file):
    rate, data = wavfile.read(wav_file)  # data : samples
    return rate, data

# Used to standardize volume of audio clip
def match_target_amplitude(sound, target_dBFS):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)

