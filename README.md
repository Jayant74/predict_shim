# Machine Learning Model for Predictive Real-Time Shimming

## Overview

This predictive model is based off Bruno and Frank's paper, ["Hybrid MRI-ultrasound acquisitions, and scannerless real-time imaging"](https://pmc.ncbi.nlm.nih.gov/articles/PMC5391319/). At the most basic understanding of their method, the Kernel Density Estimation model is an unbiased estimator that weights training output targets by a distance (aka kernel function) metric between a test input sample and a prior training input samples. 

## Machine Learning Component

One must be cautious when implementing ML in a small-dataset problem. One dataset of OCM-B0 data contains 170 dynamic B0 maps [], and 170 windows of OCM data []. Even a small, lightweight ML model can contain over 100k parameters, so it is unlikely a direct relationship between OCM data and B0 maps can be inferred, much less a generalizable function.

Therefore any ML module must be used at most as a feature extractor, or latent space extractor, to represent the data better for the KDE to do its thing.  

## Method

The machine learning model is setup as a feature extractor for the unbiased estimator, the KDE. In this method, a small CNN module acts to map OCM windows to a smaller dimensional latent space, and the distance between these latent vectors are used as weights in the KDE weighted averaging process.  

