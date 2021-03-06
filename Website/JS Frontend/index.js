document.getElementById("results").style.visibility = "hidden"

const form = document.getElementById('main-form');

form.addEventListener('submit', e => {
    e.preventDefault();

    document.getElementById("results").style.visibility = "hidden"

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

  let fetches = 0
  function didFetch() {
    fetches += 1
    console.log(fetches);
    console.log(fetches == 3);
    if (fetches == 3) {
      document.getElementById("results").style.visibility = "visible"
    }
  }

  fetch('http://127.0.0.1:5000/api/data/' + original_spectogram_name, {
    method: 'GET'
  }).then(function(resp) {
    return resp.blob();
  }).then(function(blob) {
    didFetch()
    document.getElementById("before-image").src = URL.createObjectURL(blob)
  }).catch(error => alert(error));

  fetch('http://127.0.0.1:5000/api/data/' + new_spectogram_name, {
    method: 'GET'
  }).then(function(resp) {
    return resp.blob();
  }).then(function(blob) {
    didFetch()
    document.getElementById("after-image").src = URL.createObjectURL(blob)
  }).catch(error => alert(error));

  let audio_file_url = 'http://127.0.0.1:5000/api/data/' + denoised_audio_path
  document.getElementById("download-button").onclick = function() {
    var link = document.createElement("a");
    link.download = "denoised";
    link.href = audio_file_url;
    link.click();
  }

  fetch(audio_file_url, {
    method: 'GET'
  }).then(function(resp) {
    return resp.blob();
  }).then(function(blob) {
    didFetch()
    document.getElementById("after-audio").src = URL.createObjectURL(blob)
  }).catch(error => alert(error));

}

// All image comparision code from https://www.w3schools.com/howto/howto_js_image_comparison.asp
function initComparisons() {
  var x, i;
  /*find all elements with an "overlay" class:*/
  x = document.getElementsByClassName("img-comp-overlay");
  for (i = 0; i < x.length; i++) {
    /*once for each "overlay" element:
    pass the "overlay" element as a parameter when executing the compareImages function:*/
    compareImages(x[i]);
  }
  function compareImages(img) {
    var slider, img, clicked = 0, w, h;
    /*get the width and height of the img element*/
    w = img.offsetWidth;
    h = img.offsetHeight;
    /*set the width of the img element to 50%:*/
    img.style.width = (w / 2) + "px";
    /*create slider:*/
    slider = document.createElement("DIV");
    slider.setAttribute("class", "img-comp-slider");
    /*insert slider*/
    img.parentElement.insertBefore(slider, img);
    /*position the slider in the middle:*/
    slider.style.top = (h / 2) - (slider.offsetHeight / 2) + "px";
    slider.style.left = (w / 2) - (slider.offsetWidth / 2) + "px";
    /*execute a function when the mouse button is pressed:*/
    slider.addEventListener("mousedown", slideReady);
    /*and another function when the mouse button is released:*/
    window.addEventListener("mouseup", slideFinish);
    /*or touched (for touch screens:*/
    slider.addEventListener("touchstart", slideReady);
    /*and released (for touch screens:*/
    window.addEventListener("touchstop", slideFinish);
    function slideReady(e) {
      /*prevent any other actions that may occur when moving over the image:*/
      e.preventDefault();
      /*the slider is now clicked and ready to move:*/
      clicked = 1;
      /*execute a function when the slider is moved:*/
      window.addEventListener("mousemove", slideMove);
      window.addEventListener("touchmove", slideMove);
    }
    function slideFinish() {
      /*the slider is no longer clicked:*/
      clicked = 0;
    }
    function slideMove(e) {
      var pos;
      /*if the slider is no longer clicked, exit this function:*/
      if (clicked == 0) return false;
      /*get the cursor's x position:*/
      pos = getCursorPos(e)
      /*prevent the slider from being positioned outside the image:*/
      if (pos < 0) pos = 0;
      if (pos > w) pos = w;
      /*execute a function that will resize the overlay image according to the cursor:*/
      slide(pos);
    }
    function getCursorPos(e) {
      var a, x = 0;
      e = e || window.event;
      /*get the x positions of the image:*/
      a = img.getBoundingClientRect();
      /*calculate the cursor's x coordinate, relative to the image:*/
      x = e.pageX - a.left;
      /*consider any page scrolling:*/
      x = x - window.pageXOffset;
      return x;
    }
    function slide(x) {
      /*resize the image:*/
      img.style.width = x + "px";
      /*position the slider:*/
      slider.style.left = img.offsetWidth - (slider.offsetWidth / 2) + "px";
    }
  }
}
