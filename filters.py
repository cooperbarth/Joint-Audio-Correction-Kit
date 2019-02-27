#an algorithm for applying low-pass and SD-rom filters
import numpy as np, librosa as lib

#Constants for SD-rom
WINDOW_SIZE = 5
THRESHOLD_1 = 4
THRESHOLD_2 = 12

#pass in a window of size n with the center sample under inspection
def SD_rom(window):
    #getting the sample we're looking at
    center_index = len(window) // 2
    center_sample = window[center_index] #x(n)

    #organizing the sliding window i.e. the window without the center sample
    sliding_window = window.copy() #w(n)
    del sliding_window[center_index]
    sliding_window = sorted(sliding_window) #r(n)

    #calculating the rank-ordered mean
    ROM = (sliding_window[center_index - 1] + sliding_window[center_index]) / 2

    #construct rank order differences
    rank_order_diff = []
    for i in range(len(sliding_window)):
        if center_sample > ROM:
            rank_order_diff.append(center_sample - sliding_window[-i - 1])
        else:
            rank_order_diff.append(sliding_window[i] - center_sample)
    
    #replace x(n) with ROM if impulse detected
    if rank_order_diff[0] > THRESHOLD_1 or rank_order_diff[1] > THRESHOLD_2:
        window[center_index] = ROM
        
    return window


def filter():
    print("filter here!")
