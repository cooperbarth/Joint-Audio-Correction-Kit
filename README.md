# Joint Audio Correction Kit (J.A.C.K.)
A modern solution for audio noise reduction and active noise cancellation.

![marketing image](https://github.com/cooperbarth/Joint-Audio-Correction-Kit/raw/master/Images/MacBookPro.png "Marketing Image")

## Motivation
- Implement a useful feature from audio editing software
- Understand noise reduction and how to isolate sources from a single sound
- Explore potential alternatives to current implementations of noise reduction and cancellation

## Existing Approaches
- Different algorithms are needed depending on the type of sound to be eliminated, such as wind, rain, and background conversation
- Stationary noise, or noise that does not change in nature, is commonly removed using Wiener Filtering or Nonnegative Matrix Factorization.
- A noise estimate can be created from a portion of the signal containing only background.
- Effectiveness of an algorithm is based on how it maximizes a sample-to-noise ratio, SNR, where the signal is compared to the noise estimate after the signal has been noise reduced

## Obstacles
- Preserving the original signal quality so speech is easily understandable
- Properly identifying the background noise
- Implementing an algorithm that works for most examples of the selected noise type
- Noise artifacts after completing noise reduction

## Approach
- To filter the noisy audio, an adaptive Wiener filtering method was implemented.
- The filter utilizes a two-step noise removal and signal repair method.
![approach](https://github.com/cooperbarth/Joint-Audio-Correction-Kit/raw/master/Images/Approach.png "Approach")

## Testing
(specifically, how do we know it's good)

## Results

### Before:
![before](https://github.com/cooperbarth/Joint-Audio-Correction-Kit/raw/master/Images/before.png "before")

### After:
![after](https://github.com/cooperbarth/Joint-Audio-Correction-Kit/raw/master/Images/after.png "after")

## Installation Instructions
1. git clone `https://github.com/cooperbarth/Joint-Audio-Correction-Kit`
2. cd into `/Website/Python Server/`
4. install the required packages `conda create -n eecs352p --file requirements.txt`
5. to start the local server run `FLASK_APP=server.py FLASK_DEBUG=1 python -m flask run`
6. cd into `/Website/JS Frontend/`
7. open `index.html`

Created by Cooper Barth, Andrew Finke, and Jack Wiig for EECS 352 - Machine Perception of Music and Audio @ Northwestern University


https://cooperbarth.github.io/Joint-Audio-Correction-Kit/
