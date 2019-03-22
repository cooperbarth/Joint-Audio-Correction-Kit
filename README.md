# Joint Audio Correction Kit (J.A.C.K.)
A modern solution for audio noise reduction and active noise cancellation. Created by Cooper Barth (cooperbarth2021@u.northwestern.edu), Andrew Finke, and Jack Wiig for EECS 352 - Machine Perception of Music and Audio @ Northwestern University with Professor Pardo.

![marketing image](https://github.com/cooperbarth/Joint-Audio-Correction-Kit/raw/master/Resources/MacBookPro.png)

## Motivation
 Our goal was to implement a useful feature from audio editing software. Additionaly, we sought to better understand how noise reduction was implemented and how to isolate sources from a single sound. We also wanted to explore potential alternatives to current implementations of noise reduction and cancellation.

## Background
  Currently, different algorithms are needed depending on the type of sound to be eliminated, such as wind, rain, and background conversation. Stationary noise, or noise that does not change in nature, is commonly removed using Wiener Filtering or Nonnegative Matrix Factorization. One approach to perform this task is to use noise estimate created from a portion of the signal containing only background. The effectiveness of an algorithm that attempts to reduce noise is based on how it maximizes a sample-to-noise ratio, SNR, where the signal is compared to the noise estimate after the signal has been noise reduced.

## Obstacles
  Removing noise without the original source files presents numerous obstacles. Firstly, preserving the original signal quality so speech is easily understandable is one such challenge. Additionally, properly identifying the background noise is key as an improper classification could lead to the source signal being removed or not a sufficiently level of noise reduced. Because of this, implementing an algorithm that works for most examples of the selected noise type is a difficult undertaking as there are many different types of noise. Using an algorithm designed for a different type of noise than the one present in the input source file could lead to sporadic results. Finally, after completing noise reduction, noise artifacts may still be left over in the output track.

## Approach
  To filter the noisy audio, we choose to use an adaptive Wiener filtering method to reduce the noise in our source audio tracks. The filter utilizes a two-step noise removal and signal repair method, as outlined in the following figure:
![approach](https://github.com/cooperbarth/Joint-Audio-Correction-Kit/raw/master/Resources/Approach.png "Approach")

## Testing
(specifically, how do we know it's good)

## Results

### Before:
[Initial Audio](https://github.com/cooperbarth/Joint-Audio-Correction-Kit/raw/master/Resources/buble_with_noise.wav)
![before](https://github.com/cooperbarth/Joint-Audio-Correction-Kit/raw/master/Resources/before.png "before")
- Some text explaining this

### After:
[Processed Audio](https://github.com/cooperbarth/Joint-Audio-Correction-Kit/raw/master/Resources/buble_without_noise.wav)
![after](https://github.com/cooperbarth/Joint-Audio-Correction-Kit/raw/master/Resources/after.png "after")
- Some text explaining this

## Installation Instructions
1. git clone `https://github.com/cooperbarth/Joint-Audio-Correction-Kit`
2. cd into `/Website/Python Server/`
4. install the required packages `conda create -n eecs352p --file requirements.txt`
5. to start the local server run `FLASK_APP=server.py FLASK_DEBUG=1 python -m flask run`
6. cd into `/Website/JS Frontend/`
7. open `index.html`



https://cooperbarth.github.io/Joint-Audio-Correction-Kit/
