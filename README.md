# GFit
GFit is a fitting program that fits any spectrum with Gaussians without giving any initial parameters.
That is, the spectrum is regarded as a frequency distribution (Gaussian mixture model), and then some clustering methods are performed. 
This program can also perform "peak extraction" based on the peak height of each Gaussian relative to the volume of noise in the spectrum.

## Latest version
Version 1.0 (2022 Feb 02)

## Changes
* version 1.0
    - k-means clustering method
    - EM algorithm
    - Variational Bayes method

## Required Packages
- python >= 3.7
- matplotlib
- numpy
- scipy (for digamma function in variational Bayes calculation)

## Manual

## License
GFit is distributed under the MIT License.
Copyright (c) 2022 GFit Development Team
