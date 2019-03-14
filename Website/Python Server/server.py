from os.path import join, exists
from os import mkdir
import logging

import config
from flask import Flask, request, redirect, send_file, jsonify
from plotting import plot_audio, plt_spectrogram

from placeholder import this_is_going_to_totally_work_right
from save_audio import wavwrite, process_request

app = Flask(__name__)


@app.route('/')
def hmm():
    return redirect("https://www.youtube.com/watch?v=25QyCxVkXwQ")


# https://stackoverflow.com/questions/34066804/disabling-caching-in-flask
@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r


@app.route('/api/data/<filename>', methods=['GET'])
def fetch_file(filename):
    app.logger.info("carrying the team x 2")
    app.logger.info("grabbing " + str(filename))
    return send_file(config.API_TEMP_DIRECTORY + filename)


@app.route('/api/file_denoiser', methods=['POST'])
def file_denoiser():
    app.logger.info("file_denoiser: loading file")
    signal, sample_rate, filename = process_request(request)

    app.logger.info("file_denoiser: plotting file")
    window_size = int(512)

    original_spectogram_path = plt_spectrogram(
        signal,
        window_size,
        int(window_size / 2),
        sample_rate,
        filename=filename)

    denoised_signal = this_is_going_to_totally_work_right(signal, sample_rate)

    new_spectogram_path = plt_spectrogram(
        denoised_signal,
        window_size,
        int(window_size / 2),
        sample_rate,
        filename="new-" + filename)

    denoised_audio_path = config.API_TEMP_DIRECTORY + filename + '.wav'
    wavwrite(denoised_audio_path, denoised_signal, sample_rate)

    app.logger.info("file_denoiser: sending json")
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
