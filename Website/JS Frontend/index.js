const form = document.querySelector('form');

form.addEventListener('submit', e => {
    e.preventDefault();

    var file = document.querySelector('input[type="file"]').files[0];
    const formData = new FormData();
    formData.append('audio_file', file);

    fetch('http://127.0.0.1:5000/api/file_to_waveform_image', {
      method: 'POST',
      body: formData
    }).then(function(resp) {
      return resp.blob();
    }).then(function(blob) {
      document.querySelector('img').src = URL.createObjectURL(blob);
    }).catch(error => alert(error));
});
