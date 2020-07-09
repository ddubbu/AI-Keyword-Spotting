from pydub import AudioSegment
import random
import sys
import io
import os
import glob
# import IPython
import numpy as np
import matplotlib.pyplot as plt
from wav_tools import *  # 빨간 밑줄 상관 없음.
from generate_train_data import *

'''
1. Listening to the data : 그냥 self로 들음. 
    Jupyter notebook 말고 여기서 들을 수 있는 방법은?
'''


'''
2. From audio recordings to spectrograms

<key values>
We will train with 10s data in each case.
- n_freq : # of frequencies input at each time step
- raw audio : audio sampled at 441000Hz
- Tx : # of time steps from spectrogram
        I think, even if Tx n_freq is 441000 Hz,
        each step 
            10/441000 = 0.000023s 씩
            (V) 10/5511 = 0.0015s 씩
        second one 으로도 충분히 커버되어서 그러는 듯.
- Ty : # output of the GRU (10s into 1375 time steps, each step is 0.0072s) 
        we will check ach intervals whether someone recently finished saying "activate"(or "something")
'''
x = graph_spectrogram("audio_examples/example_train.wav")  # x.shape = (101, 5511)
_, data = wavfile.read("audio_examples/example_train.wav")  # data[:,0].shape = (441000, )

n_freq, Tx = x.shape[0], x.shape[1]  # dimension of input to NN
Ty = 1375


'''
3. generating a single training example
we will make training data 
by synthesizing activates, negatives(not activates), backgrounds(noise)
which are 0~4s, 0~2s, 10s

+ 어차피 background에 insert 되는거라 상관없을 거 같긴 한데, 
activate/negative sounds doesn't have silence.

Q. How to insert? 
A. simple by pydub.overlay function
    randomly pick the each data 
    overlaying positive/negative words onto the background.
    So, it is 10s based backgrounds data.
    
    내맘대로 넣기도, 바꾸기도 하면서 다양한 training data 만들 수 있겠다.

★ ★ ★
나는, y labeling을 단순히 keyword 존재 여부 (0/1)가 아니라,
정확히 어떤 keyword가 / 어디에 있는지까지
2가지 정보를 학습하고 싶고

Q. y output=1 50 step인 이유는?? 
    y가 무엇이기에.. 0~1400 까지 x-axis를 갖고 있는가..
'''

# Load audio segments using pydub
activates, negatives, backgrounds = load_raw_audio()

# Create training example

# x = graph_spectrogram("train.wav")


# x, y = create_training_example(Ty, backgrounds, activates, negatives, num)  # num : 파일 번호
#
# print(y)
# plt.figure()
# plt.plot(y[0])
# plt.show()

# make x,y dataset
create_training_examples(Ty, 20)

'''
4. Full Training set
Load preprocessed training examples
'''
# X = np.load("./XY_train/X.npy")