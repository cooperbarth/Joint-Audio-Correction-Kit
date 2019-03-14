const form = document.getElementById('main-form');

form.addEventListener('submit', e => {
    e.preventDefault();

    var file = document.getElementById('audio_file').files[0];
    const formData = new FormData();
    formData.append('audio_file', file);

    const blob = new Blob([file]);
    document.getElementById("before-audio").src = URL.createObjectURL(blob)

    let endpoint = "file_denoiser"
    fetch('http://127.0.0.1:5000/api/' + endpoint, {
      method: 'POST',
      body: formData
    }).then(function(response) {
      return response.json();
    }).then(function(response) {
      fetchData(response)
      console.log(JSON.stringify(response));
    }).catch(error => alert(error));
});

function fetchData(results) {
  let original_spectogram_name = results["original_spectogram_name"]
  let new_spectogram_name = results["new_spectogram_name"]
  let denoised_audio_path = results["denoised_audio_path"]

  fetch('http://127.0.0.1:5000/api/data/' + original_spectogram_name, {
    method: 'GET'
  }).then(function(resp) {
    return resp.blob();
  }).then(function(blob) {
    document.getElementById("before-image").src = URL.createObjectURL(blob)
  }).catch(error => alert(error));

  fetch('http://127.0.0.1:5000/api/data/' + new_spectogram_name, {
    method: 'GET'
  }).then(function(resp) {
    return resp.blob();
  }).then(function(blob) {
    document.getElementById("after-image").src = URL.createObjectURL(blob)
  }).catch(error => alert(error));

  fetch('http://127.0.0.1:5000/api/data/' + denoised_audio_path, {
    method: 'GET'
  }).then(function(resp) {
    return resp.blob();
  }).then(function(blob) {
    document.getElementById("after-audio").src = URL.createObjectURL(blob)
  }).catch(error => alert(error));

}
