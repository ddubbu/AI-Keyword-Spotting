'''
This is for PART 3

3. generating a single training example
we will make training data
by synthesizing activates, negatives(not activates), backgrounds(noise)
which are 0~4s, 0~2s, 10s
'''

import numpy as np
import os
from pydub import AudioSegment  # for manipulating audio data.
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
from wav_tools import *

# Load raw audio files for speech synthesis
def load_raw_audio():
    activates = []
    backgrounds = []
    negatives = []
    for filename in os.listdir("./raw_data/activates"):
        if filename.endswith("wav"):
            activate = AudioSegment.from_wav("./raw_data/activates/"+filename)
            activates.append(activate)
    for filename in os.listdir("./raw_data/backgrounds"):
        if filename.endswith("wav"):
            background = AudioSegment.from_wav("./raw_data/backgrounds/"+filename)
            backgrounds.append(background)
    for filename in os.listdir("./raw_data/negatives"):
        if filename.endswith("wav"):
            negative = AudioSegment.from_wav("./raw_data/negatives/"+filename)
            negatives.append(negative)
    return activates, negatives, backgrounds

def get_random_time_segment(segment_ms):
    """
    find random position for len(segment_ms) in a 10,000 ms(=10s) audio clip.
    segment_ms -- the duration of the audio clip in ms
        ★ Q. activate/negative data 의 길이인가?
           A. 그런듯.
    """

    # I change : low : 0 -> segment_ms
    # 왜냐하면, 음수 시간에서도 시작하니깐 안됨.
    segment_start = np.random.randint(low=segment_ms, high=10000 - segment_ms)  # Make sure segment doesn't run past the 10sec background
    segment_end = segment_start + segment_ms - 1

    return (segment_start, segment_end)


def is_overlapping(new_segment, previous_segments):
    """
    Checks if the time of a segment overlaps with the times of existing segments.
    previous_segments could be many.
    """

    segment_start, segment_end = new_segment
    # Step 1: Initialize overlap as a "False" flag.
    overlap = False

    # Step 2: loop over the previous_segments start and end times.
    for previous_start, previous_end in previous_segments:  # it has a tuple data.
        if segment_start <= previous_end and segment_end >= previous_start:
            overlap = True

    return overlap  # Bool


def insert_audio_clip(background, new_audio_clip, previous_segments):
    """
    Insert a new audio segment over the background noise at a random time step,
    ensuring that new audio segment does not overlap with existing segments.
    """

    # Get the duration of the audio clip in ms
    segment_ms = len(new_audio_clip)

    # Step 1: Pick a random time segment onto which to insert new audio clip.
    segment_time = get_random_time_segment(segment_ms)

    # Step 2: Check if overlap with previous segments.
    # ★★★  Warning : Infinite Loop ★★★
    while is_overlapping(segment_time, previous_segments):
        segment_time = get_random_time_segment(segment_ms)

    # Step 3: Add previous list not to overlap
    previous_segments.append(segment_time)

    # Step 4: Insert audio segments and background
    new_background = background.overlay(new_audio_clip, position=segment_time[0])

    return new_background, segment_time


def insert_ones(Ty, y, segment_end_ms):
    """
    Update the label vector y. The labels of the 50 output steps strictly after the end of the segment
    should be set to 1. By strictly we mean that the label of segment_end_y should be 0 while, the
    50 following labels should be ones.

    ★ ★ Q. Why?  50 labels??
        len(new_segment) 만큼 아님?
        그리고 왜 끝나는 지점부터 50 segment를 하냐고....


    Arguments:
    y -- numpy array of shape (1, Ty), the labels of the training example
    segment_end_ms -- the end time of the segment in ms

    Returns:
    y -- updated labels
    """

    # duration of the background (in terms of spectrogram time-steps)
    segment_end_y = int(segment_end_ms * Ty / 10000.0)

    # Add 1 to the correct index in the background label (y)
    for i in range(segment_end_y + 1, segment_end_y + 50 + 1):
        if i < 1375:
            y[0, i] = 1

    return y


# GRADED FUNCTION: create_training_example

def create_training_example(Ty, backgrounds, activates, negatives):
    """
    Creates a training example with a given background, activates, and negatives.

    Arguments:
    background -- a 10 second background audio recording
    activates -- a list of audio segments of the word "activate"
    negatives -- a list of audio segments of random words that are not "activate"

    Returns:
    x -- the spectrogram of the training example
    y -- the label at each time step of the spectrogram
    """

    # Set the random seed
    # np.random.seed(18) ^^ 너때문에 random은 커녕 계속 같은 값이 나오잖니?

    # Step 1: Initialize y (label vector) of zeros (≈ 1 line)
    y = np.zeros((1, Ty))

    # Step 2: Initialize segment times as empty list (≈ 1 line)
    previous_segments = []

    # Step 3: # Select 0-1 random "background" audio clips from the entire list of "backgrounds" recordings
    random_indices = np.random.randint(len(backgrounds))
    print(random_indices)
    background = backgrounds[random_indices]
    # Make background quieter
    background = background - 20  # ★ 왜?

    # Select 0-4 random "activate" audio clips from the entire list of "activates" recordings
    number_of_activates = 1  # ★★★ 한개만 넣자  //np.random.randint(0, 5)
    random_indices = np.random.randint(len(activates), size=number_of_activates)
    random_activates = [activates[i] for i in random_indices]

    # Step 4: Loop over randomly selected "activate" clips and insert in background
    for random_activate in random_activates:
        # Insert the audio clip on the background
        background, segment_time = insert_audio_clip(background, random_activate, previous_segments)
        print("activate insert time :", segment_time)
        # Retrieve segment_start and segment_end from segment_time
        segment_start, segment_end = segment_time
        # Insert labels in "y"
        y = insert_ones(Ty, y, segment_end)

    # Select 0-2 random negatives audio recordings from the entire list of "negatives" recordings
    number_of_negatives = 1  # ★★★ 얘도 한개만 넣자. //np.random.randint(0, 3)
    random_indices = np.random.randint(len(negatives), size=number_of_negatives)
    random_negatives = [negatives[i] for i in random_indices]

    # Step 5: Loop over randomly selected negative clips and insert in background
    for random_negative in random_negatives:
        # Insert the audio clip on the background
        background, n_segment_time = insert_audio_clip(background, random_negative, previous_segments)
        print("neagtive insert time :", n_segment_time)
    # Standardize the volume of the audio clip
    background = match_target_amplitude(background, -20.0)  # ★

    # Export new training example
    file_name = "./generate_data/train" + ".wav"
    file_handle = background.export(file_name, format="wav")
    print("File was saved in ./generate_data directory.")

    # Get and plot spectrogram of the new recording (background with superposition of positive and negatives)
    x = graph_spectrogram(file_name)

    return file_name, y  # x,