# Gaussian Spectra Analysis
This project was created in order to satisfy the need for large scale automatic
analysis of peaks in gamma and x-ray spectra from nuclear reaction studies.

It can be divided into two sections, presented here as a single Python file.

___DISCLAIMER:__ The program was created attending to the specific needs of the
data being analysed. Although it may be used to analyse almost any spectra some
adjustments must be made and as such caution is advised when applying the
program to any dataset._

### 1. Analysing radiation spectra using Gaussian functions
The program will run through the spectra, find and adjust a gaussian function to
each of the peaks present. If needed, background count subtraction will be made
based on an exponential function (with parameters individually adjusted for each
of the analysed spectra).

### 2. Calculating nuclear reaction cross-sections
After analysing all peaks present in a spectrum, the program will use the
acquired information, along with specific element parameters (kinematic factor
and estimated peak positions), to calculate the excitation function of a specific
nuclear reaction.