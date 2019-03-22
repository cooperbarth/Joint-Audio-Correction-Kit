"""PyAudio use inspired by:
    https://gist.github.com/aflaxman/6300595
"""

import numpy as np, scipy.signal, pyaudio

np.seterr(divide='ignore', invalid='ignore')

WIDTH = 4
NUM_CHANNELS = 1
DEFAULT_SR = 44100
CHUNK = 2048
FILTER_LEN = 100
DTYPE = np.int16

# Create stream
p = pyaudio.PyAudio()
stream = p.open(format=p.get_format_from_width(WIDTH),
                channels=NUM_CHANNELS,
                rate=DEFAULT_SR,
                input=True,
                output=True,
                frames_per_buffer=CHUNK)

# noise_filter = np.fft.fft(np.random.rand(2*CHUNK))
filter_count = 0

# have an array of 1000 noise filters
noise_filter = np.ndarray((FILTER_LEN, 2*CHUNK), dtype=DTYPE)

# every loop, update buffer and multiply input by inverse of buffer
while True:
    # getting fft of current input chunk
    audio_input = np.frombuffer(stream.read(CHUNK), dtype=DTYPE)
    MAX_INT = max(audio_input)
    if MAX_INT == 0:
        continue
    normalized_audio_input = audio_input / MAX_INT
    freq_input = np.fft.fft(normalized_audio_input)

    # update buffer with weighted average
    if filter_count < FILTER_LEN:
        noise_filter[filter_count] = audio_input
        filter_count += 1
    else:
        noise_filter = np.append([noise_filter], audio_input[0:-1])

    filter_average = np.mean(noise_filter, axis=0)
    filter_inverse = -1 * filter_average

    # multiplier should be inverse of buffer
    freq_generated = freq_input * filter_inverse
    cancel_sound = np.real(np.fft.ifft(freq_generated))

    # output sound
    write_data = np.array(np.round(cancel_sound * MAX_INT), dtype=DTYPE) * (10 ** -10)
    stream.write(write_data.tostring(), CHUNK)

stream.stop_stream()
stream.close()
p.terminate()
