const form = document.getElementById('main-form');

form.addEventListener('submit', e => {
  console.log(432);
    e.preventDefault();

    var file = document.getElementById('audio_file').files[0];
    const formData = new FormData();
    formData.append('audio_file', file);

    let action = document.getElementById('action').value;
    let endpoint = ""
    console.log(action);
    if (action == "waveform") {
      endpoint = "file_to_waveform_image"
    } else if (action == "spectrogram") {
      endpoint = "file_to_spectrogram_image"
    } else {
      return;
    }

    fetch('http://127.0.0.1:5000/api/' + endpoint, {
      method: 'POST',
      body: formData
    }).then(function(resp) {
      return resp.blob();
    }).then(function(blob) {
      // document.querySelector('img').src = URL.createObjectURL(blob);
      document.getElementById("results").innerHTML += "<img id=result src='" + URL.createObjectURL(blob) + "'>"
    }).catch(error => alert(error));
});
