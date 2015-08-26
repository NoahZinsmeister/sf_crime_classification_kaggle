# San Francisco Crime Classification Challenge - Kaggle
This is my attempt to solve the ongoing Kaggle challenge: San Francisco Crime Classification. `main.py` is written in Python3, dependencies include `os, pickle, csv, re, numpy, matplotlib.pyplot, and sklearn`.

## Random Forest Classification
I use the [scikit learn](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) implementation of a [Random Forest Classifier](https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm) to classify the S.F. crime data. I transform the training data in `train.csv` into a 878049 x 25 (n_samples x features) feature matrix and a 878049 x 1 (n_samples x class label) class label vector.

The features are:
1. Year
2. Month
3. Day
4. Hour
5. Minute
6. Second
7. Longitude
8. Latitude
9-16. One-hot encoded variables for day of the week
17-25. One-hot encoded variables for police department district

The 39 unique class labels are each assigned integers from 0 to 38.

Using these data I train a classifier and use it to make class probability estimates on the `test.csv` data. Since my computer only has 8GB RAM, I can't do any significant cross validation without running out of memory, so for now I am training a single RF classifier with 30 trees (all trained on different bootstrapped samples), each considering a maximum of 3 features per split. If I could, I would train on many more combinations of these parameters using 4-fold CV. I create a `CV_plot.png` which plots CV log-loss scores against different parameter combinations.

My best result using this code was a log-loss score of 3.14357, putting me in 304th place out of 440 (the scores decay exponentially, so it's not quite as bad as it seems). I'm confident I could do better with more CV and some tweaks.
