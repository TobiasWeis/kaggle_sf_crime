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
        train, test, Y, Y_test = train_test_split(df_x_shuf, df_y_shuf, train_size=0.9, random_state=1337)
    len(train)
    len(np.unique(Y))
else:
    train = df_x_shuf
    Y = df_y_shuf
    test = df_x_test
    

####################
# train a classifier
####################
# T = Tree
# F = Forest
# X = xgboost

classifier = 'T'

if classifier == 'T':
    print "Training Decision Tree"
    from sklearn import tree
    # parameters obtained by random search
    clf = tree.DecisionTreeClassifier(max_depth=8, min_samples_leaf=1)
    clf.fit(train, Y)
elif classifier == 'F':
    print "Training Random Forest"
    from sklearn.ensemble import RandomForestClassifier

    # this one perfomred really good (manually set)
    '''
    clf = RandomForestClassifier(
            max_depth=16, 
            n_estimators=512, 
            n_jobs=60, 
            bootstrap=True,
            min_samples_leaf=2,
            oob_score=True)
    '''
    # this one i got from the randomized search
    clf = RandomForestClassifier(
            max_depth=20,
            n_estimators=512,
            #min_samples_split=1,
            #criterion="gini",
            #max_features=3,
            bootstrap=False,
            #min_samples_leaf=4,
            n_jobs=60, random_state=1337)
elif classifier == 'X':
    print "Training xgboost"
    import xgboost as xgb
    clf = xgb.XGBClassifier(max_depth=20, n_estimators=512, learning_rate=0.05)

st = time.time()
clf.fit(train, Y)
time_taken = time.time() - st
print "Took: ", time_taken

print "Predicting %d samples" % (len(test))
st = time.time()
preds = clf.predict_proba(test)
print "Took: ", time.time() - st

if not _submission:
    from sklearn.metrics import log_loss
    ll = log_loss(Y_test, preds)
    print "LogLoss: %f" % ll
    # write result to file for later comparison
    import datetime
    now = datetime.datetime.now()

    fname = ""
    if classifier == 'T':
        fname = "log_tree.txt"
    elif classifier == 'F':
        fname = "log_forest.txt"
    elif classifier == 'X':
        fname = "log_xgboost"

    f = open(fname, "a+")
    f.write("%02d.%02d.%d %02d:%02d ========================================\r\n" % (now.day, now.month, now.year, now.hour, now.minute))

    for k,v in clf.get_params().iteritems():
        f.write("%s:\t " % k)
        f.write(str(v))
        f.write("\r\n")

    f.write("Features: ")
    for feat in features:
        f.write("%s, " % feat)
    f.write("\r\n")
    f.write("LogLoss: %f\r\n" % ll) 
    f.write("Time taken: %f\r\n" % (time_taken) )
    f.close()
else:
    #Write results
    result=pd.DataFrame(preds, columns=clf.classes_)
    result.to_csv('testResult.csv', index = True, index_label = 'Id' )


# disabled, does not work on server
# play a sound when done!
#Playsound('./utils/complete.wav')

print "Done."
