#!/usr/bin/python

import numpy as np
import pandas as pd
from utils.playsound_tobi import *
from utils.loader import *
import sys
import time

####################
# load and preprocess data
####################
l = Loader("./data/train.csv")
df, features = l.process()

####################
# split data
####################
from sklearn.cross_validation import train_test_split

df_y = np.array(df.Category.astype('category').cat.codes.astype('float32'))
df_x = np.array(df[features]).astype('float32')

from sklearn.utils import shuffle
df_x_shuf, df_y_shuf = shuffle(df_x, df_y, random_state=0)

Y_test = []
Y = [1,2,3]
# log_loss only works when all categories are present in dataset
# with too small train/test sizes, sometimes a lable is not present in either group
while len(np.unique(Y_test)) != len(np.unique(Y)):
    print "Splitting dataset"
    train, test, Y, Y_test = train_test_split(df_x_shuf, df_y_shuf, train_size=0.8)
len(train)
len(np.unique(Y))

####################
# train a classifier
####################
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss

print "Training Random Forest w/ random search"
#st = time.time()
clf = RandomForestClassifier(max_depth=16, n_estimators=1024)
#clf.fit(train, Y)
#print "Took: ", time.time() - st

from scipy.stats import randint as sp_randint
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV

# Utility function to report best scores
def report(grid_scores, n_top=3):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")


# specify parameters and distributions to sample from
param_dist = {"max_depth": [3, None],
              "max_features": sp_randint(1, len(features)),
              "min_samples_split": sp_randint(1, 11),
              "min_samples_leaf": sp_randint(1, 11),
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

# run randomized search
n_iter_search = 20
random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                   n_iter=n_iter_search, n_jobs=60)

start = time.time()
random_search.fit(train, Y)
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time.time() - start), n_iter_search))
report(random_search.grid_scores_)


n_tests = len(Y_test)
print "Predicting %d samples" % (n_tests)
st = time.time()
preds = clf.predict_proba(test[:n_tests])
print log_loss(Y_test[:n_tests].astype('int'), preds)
print "Took: ", time.time() - st

# disabled, does not work on server
# play a sound when done!
#Playsound('./utils/complete.wav')

print "Done."
