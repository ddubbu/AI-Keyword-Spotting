'''
This is for PART 3

3. generating a single training example
we will make training data
by synthesizing activates, negatives(not activates), backgrounds(noise)
which are 0~4s, 0~2s, 10s
'''

import numpy as np

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