# Keypont-Detection
Key Point Detection using SIFT in pure Python


The aim of this project was to to detect keypoints in an image according to the following steps, which are also the first three
steps of Scale-Invariant Feature Transform (SIFT).

1. Generating four octaves. Each octave is composed of five images blurred using Gaussian kernels. For each
octave, the bandwidth parameters σ (five different scales) of the Gaussian kernels.
2. Computing Difference of Gaussian (DoG) for all four octaves.
3. Detecting keypoints which are located at the maxima or minima of the DoG images.  Only 
pixel-level locations of the keypoints is needed to be shown.
