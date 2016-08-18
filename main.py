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

print "Training Random Forest"
st = time.time()
clf = RandomForestClassifier(max_depth=16, n_estimators=1024, n_jobs=48)
clf.fit(train, Y)
print "Took: ", time.time() - st

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
