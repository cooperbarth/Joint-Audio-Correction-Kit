'''Code inspired by:
    https://gist.github.com/aflaxman/6300595
'''

import numpy as np, scipy.signal, pyaudio
np.seterr(divide='ignore', invalid='ignore')

WIDTH = 4
NUM_CHANNELS = 1
DEFAULT_SR = 44100
CHUNK = 2048
DTYPE = np.int16

#Create stream
p = pyaudio.PyAudio()
stream = p.open(format=p.get_format_from_width(WIDTH),
                channels=NUM_CHANNELS,
                rate=DEFAULT_SR,
                input=True,
                output=True,
                frames_per_buffer=CHUNK)

noise_filter = np.fft.fft(np.random.randn(2*CHUNK))
filter_count = 0

#every loop, update buffer and multiply input by inverse of buffer
while True:
    #getting fft of current input chunk
    audio_input = np.frombuffer(stream.read(CHUNK), dtype=DTYPE)
    MAX_INT = max(audio_input)
    if MAX_INT == 0:
        continue
    normalized_audio_input = audio_input / MAX_INT
    freq_input = np.fft.fft(normalized_audio_input)

    #update buffer with weighted average
    filter_weighted = filter_count * noise_filter
    filter_count += 1
    noise_filter = (filter_weighted + freq_input) / filter_count
    filter_inverse = -1 * noise_filter

    #multiplier should be inverse of buffer
    freq_generated = freq_input * filter_inverse
    cancel_sound = np.real(np.fft.ifft(freq_generated))

    #output sound
    write_data = np.array(np.round(cancel_sound * MAX_INT), dtype=DTYPE)
    stream.write(write_data.tostring(), CHUNK)

stream.stop_stream()
stream.close()
p.terminate()