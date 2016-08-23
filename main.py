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
df, features = l.process(do_filter=True)

l_test = Loader("./data/test.csv")
df_test, features_test = l_test.process()

####################
# split data
####################
from sklearn.cross_validation import train_test_split

#df_y = np.array(df.Category.astype('category').cat.codes.astype('float32'))
df_y = np.array(df.Category)#.astype('category')
df_x = np.array(df[features]).astype('float32')

# prepare test set
df_x_test = np.array(df_test[features]).astype('float32')

from sklearn.utils import shuffle
df_x_shuf, df_y_shuf = shuffle(df_x, df_y, random_state=0)

_submission = True

if not _submission:
    Y_test = []
    Y = [1,2,3]
    # log_loss only works when all categories are present in dataset
    # with too small train/test sizes, sometimes a lable is not present in either group
    while len(np.unique(Y_test)) != len(np.unique(Y)):
        print "Splitting dataset"
        train, test, Y, Y_test = train_test_split(df_x_shuf, df_y_shuf, train_size=0.8)
    len(train)
    len(np.unique(Y))
else:
    train = df_x_shuf
    Y = df_y_shuf
    test = df_x_test

####################
# Classification
####################
# T = tree
# F = forest
classifier = 'T'

st = time.time()
if classifier == 'T': 
    from sklearn import tree
    clf = tree.DecisionTreeClassifier()
    clf.fit(train, Y)
    print "Training Decision Tree"

if classifier == 'F':
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import log_loss

    print "Training Random Forest"
    # this one perfomred really good (manually set)
    clf = RandomForestClassifier(max_depth=16, n_estimators=1024, n_jobs=48)
    # this one i got from the randomized search
    #clf = RandomForestClassifier(max_features=2, min_samples_split=3, criterion="entropy", min_samples_leaf=3, n_estimators=1024, n_jobs=64, max_depth=None)
    clf.fit(train, Y)
print "Took: ", time.time() - st

####################
# Prediction
####################
print "Predicting %d samples" % (len(test))
st = time.time()
preds = clf.predict_proba(test)
print "Took: ", time.time() - st


#####################
# Output
#####################
if not _submission:
    print log_loss(Y_test[:n_tests].astype('int'), preds)
else:
    #Write results
    result=pd.DataFrame(preds, columns=clf.classes_)
    result.to_csv('testResult.csv', index = True, index_label = 'Id' )

# disabled, does not work on server
# play a sound when done!
#Playsound('./utils/complete.wav')

print "Done."
