from flask import Flask, request, redirect, send_file

import config

from os.path import join, exists
from os import mkdir
from flask import jsonify

from plotting import plot_audio, plt_spectrogram
app = Flask(__name__)

from placeholder import this_is_going_to_totally_work_right
from save_audio import wavwrite, process_request


@app.route('/')
def hmm():
    return redirect("https://www.youtube.com/watch?v=25QyCxVkXwQ")


@app.route('/api/data/<filename>', methods=['GET'])
def fetch_file(filename):
    print("carrying the team x 2")
    return send_file(config.API_TEMP_DIRECTORY + filename)


@app.route('/api/file_denoiser', methods=['POST'])
def file_denoiser():
    signal, sample_rate, filename = process_request(request)

    print("file_denoiser: loading file")

    print("file_denoiser: plotting file")
    window_size = int(512)

    original_spectogram_path = plt_spectrogram(
        signal,
        window_size,
        int(window_size / 2),
        sample_rate,
        filename=filename)

    denoised_signal = this_is_going_to_totally_work_right(signal, sample_rate)

    new_spectogram_path = plt_spectrogram(
        signal, window_size,
        int(window_size / 2),
        sample_rate,
        filename="new-" + filename)

    denoised_audio_path = config.API_TEMP_DIRECTORY + filename + '.wav'
    wavwrite(denoised_audio_path, denoised_signal, sample_rate)

    print("file_denoiser: sending file")
    return jsonify({
        "original_spectogram_name": original_spectogram_path.rsplit('/', 1)[-1],
        "new_spectogram_name": new_spectogram_path.rsplit('/', 1)[-1],
        "denoised_audio_path": denoised_audio_path.rsplit('/', 1)[-1]
    })


if __name__ == '__main__':
    if not exists(config.API_DIRECTORY):
        mkdir(config.API_DIRECTORY)
    if not exists(config.API_TEMP_DIRECTORY):
        mkdir(config.API_TEMP_DIRECTORY)
