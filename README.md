# Code samples

This repo contains some code samples, organized as follows:

* `custom-db` is a small pandas-based database I wrote to easily access and navigate through a dataset of ~700 electrophysiological recordings acquired under different experimental conditions.
* `theano-filters` is a symbolic implementation of (linear or non-linear) filtering for time-series objects.
* `tests` contains a series of tests written for the `maprf` package (which for the moment is still private).

There is also a folder containing some jupyter notebooks, stored in html format.
* The purpose of `0016-analysis-demo-Copy2` is to illustrate to a user how to perform an end-to-end analysis using the Bayesian inference framework I developed to identify the parameters of a model of neural activity (specifically, this is a model of a type of neurons we encounter early on in the visual system).
* `make_figure_2` is code to make a figure to inspect in higher details the results of an analysis such as the one illustrated in `0016-analysis-demo-Copy2`.
* `validation_analysis_collapsed_ns` analyzes the performance of an sampling algorithm I developed to improve the efficiency of a well-known sampling algorithm (nested sampling) when used on a particular class of models (GLMs). Its performance are compared against those of normal nested sampling. Specifically, it compares performances both in computational terms (time and memory) and in terms of accuracy of the estimates.
