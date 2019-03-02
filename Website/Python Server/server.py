from flask import Flask, request, redirect

import config
import waveforms

app = Flask(__name__)


@app.route('/')
def hmm():
    return redirect("https://www.youtube.com/watch?v=25QyCxVkXwQ")


@app.route('/api/file_to_waveform_image', methods=['POST'])
def file_to_waveform_image():
    if not request.files:
        return "error"
    file = None
    for key, file_storage in request.files.items():
        if key == "audio_file":
            file = file_storage

    if not file:
        print("file_to_waveform_image: no file")
        return "error"
    return waveforms.file_to_waveform_image(file)
