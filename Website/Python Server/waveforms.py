from os.path import join

from flask import send_file
from werkzeug.utils import secure_filename
from librosa import load

from plotting import plot_audio
import config


def file_to_waveform_image(file):
    print("file_to_waveform_image: started")
    if file and file.filename.endswith(".mp3"):
        filename = secure_filename(file.filename)
        path = join(config.API_TEMP_DIRECTORY, filename)
        print("file_to_waveform_image: saving file")
        file.save(path)

        print("file_to_waveform_image: loading file")
        signal, sample_rate = load(path)

        print("file_to_waveform_image: plotting file")
        image_path = plot_audio(signal, sample_rate, filename)

        print("file_to_waveform_image: sending file")
        return send_file(image_path,
                         mimetype='image/png',
                         attachment_filename='result.png',
                         as_attachment=True)

    print("file_to_waveform_image_form: bad file type")
    return "error"
