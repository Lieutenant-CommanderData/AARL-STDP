import numpy as np
from random import randint

'''
Inter Spike Interval (ISI) Function

Calculates the frequncy of spikes in the input array

Returns the frequency and the time at which that frequency occured
'''
def getFreq(data, dt):
    loc = np.where(data == 1)[0]
    freq = []
    for i in range(len(loc)-1):
        freq.append(1 / ((loc[i+1] - loc[i]) * (dt * 0.001)))
    # The first element needs to be removed because this is inner spike, so more spikes than frequencies recorded
    loc = loc[1:] * dt
    return freq, loc

'''
Smooth to Noisy Signal Generator
Takes your smooth signal, makes it noisy
'''
def NoisyAmps(smooth, percent_noise=float):
    percent_noise = percent_noise / 100
    noisy = np.copy(smooth)
    time, motor = smooth.shape
    for mn in range(motor):
        for i in range(time):
            if smooth[i, mn] > 1.0:
                noisy[i, mn] = randint(int((smooth[i, mn] * (1 - percent_noise)) * 10000), int((smooth[i, mn] * (1 + percent_noise)) * 10000)) / 10000
    return noisy
