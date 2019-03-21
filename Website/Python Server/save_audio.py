import scipy.io.wavfile
import numpy as np
import config

from flask import send_file, jsonify
from werkzeug.utils import secure_filename
from librosa import load
from os.path import join, exists


def wavwrite(filepath, data, sr, norm=True, dtype='int16',):
    '''
    Write wave file using scipy.io.wavefile.write, converting from a float (-1.0 : 1.0) numpy array to an integer array

    Parameters
    ----------
    filepath : str
        The path of the output .wav file
    data : np.array
        The float-type audio array
    sr : int
        The sampling rate
    norm : bool
        If True, normalize the audio to -1.0 to 1.0 before converting integer
    dtype : str
        The output type. Typically leave this at the default of 'int16'.
    '''
    if norm:
        data /= np.max(np.abs(data))
    data = data * np.iinfo(dtype).max
    data = data.astype(dtype)
    scipy.io.wavfile.write(filepath, sr, data)


def process_request(request):
    if not request.files:
        return
    file = None
    for key, file_storage in request.files.items():
        if key == "audio_file":
            file = file_storage

    if not file:
        app.logger.error("process_request: no file")
        return

    if not file.filename.endswith(".wav"):
        app.logger.error("not wav")
        return

    filename = secure_filename(file.filename).split(".")[0]
    path = join(config.API_TEMP_DIRECTORY, filename)
    file.save(path)

    signal, sample_rate = load(path)
    return signal, sample_rate, filename
