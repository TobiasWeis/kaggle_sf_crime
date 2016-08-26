#!/usr/bin/python

import numpy as np
import pandas as pd
from utils.playsound_tobi import *
from utils.loader import *
import sys
import time
from operator import itemgetter

####################
# load and preprocess data
####################
l = Loader("./data/train.csv")
df, features = l.process()

####################
# split data
####################
from sklearn.cross_validation import train_test_split

'''
Y_test = []
Y = [1,2,3]
# log_loss only works when all categories are present in dataset
# with too small train/test sizes, sometimes a lable is not present in either group
while len(np.unique(Y_test)) != len(np.unique(Y)):
    print "Splitting dataset"
    train, test, Y, Y_test = train_test_split(df_x_shuf, df_y_shuf, train_size=0.7)
len(train)
len(np.unique(Y))
'''

####################
# train a classifier
####################
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss

def scorer(estimator,x,y):
    preds = estimator.predict_proba(x)
    # smaller values are better!
    return -log_loss(y, preds)

# Utility function to report best scores
def report(grid_scores, features_used, n_top=3, clfname="NOT_GIVEN"):
    import datetime
    now = datetime.datetime.now()

    f = open("log_search_output.txt", "a+")
    f.write("%02d.%02d.%d %02d:%02d == %s ======================================\r\n" % (now.day, now.month, now.year, now.hour, now.minute, clfname))
    f.write("Features used: ")
    for feat in features_used:
        f.write(feat)
        f.write(", ")
    f.write("\r\n")
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        f.write("Model with rank: {0}\r\n".format(i + 1))
        print("Model with rank: {0}".format(i + 1))
        f.write("Mean validation score: {0:.3f} (std: {1:.3f})\r\n".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        f.write("Parameters: {0}\r\n".format(score.parameters))
        print("Parameters: {0}".format(score.parameters))
        print("")
    f.close()


classifier = 'T'

from scipy.stats import randint as sp_randint
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV

for classifier in ['T','F','A']:
    print "======================================= Training %s" % classifier
    clfname = ""
    for features in [
            ["DayOfWeek","PdDistrict_num", "Hour"],
            ["X","Y","DayOfWeek","Hour"],
            ["X","Y","DayOfWeek","PdDistrict_num","Hour"],
            ["X","Y","DayOfWeek","PdDistrict_num","Hour","Month","Year","DayOfYear"]
            ]:
        print "=================================== Features"
        print features
        print "==================================="
        df_y = np.array(df.Category.astype('category').cat.codes.astype('float32'))
        df_x = np.array(df[features]).astype('float32')

        from sklearn.utils import shuffle
        df_x_shuf, df_y_shuf = shuffle(df_x, df_y, random_state=0)


        if classifier == 'T':
            clfname = "DecisionTree"
            from sklearn import tree
            clf = tree.DecisionTreeClassifier()
            param_dist = {"max_depth": [3,6,9,12,15,20,25,30,35,40,45,50,100,125,150,175,200],
                          "min_samples_leaf" : [1,2,3,4,5,10,15,20,25,40,60,80,100],
                          "class_weight": ["balanced", None],
                          "max_features": sp_randint(1, len(features)),
                          "min_samples_split": sp_randint(1, len(features))
                          }
        elif classifier == 'F':
            clfname = "RandomForest"
            print "Training Random Forest w/ random search"
            clf = RandomForestClassifier(n_jobs=50)
            # specify parameters and distributions to sample from
            param_dist = {"max_depth": [8,12,16,20,32,64],
                          "n_estimators": [128,256, 512, 1024, 2048,4096],
                          "max_features": sp_randint(1, len(features)),
                          "min_samples_split": sp_randint(1, len(features)),
                          "min_samples_leaf": sp_randint(1, len(features)),
                          "bootstrap": [True, False],
                          "criterion": ["gini", "entropy"]}
        elif classifier == 'A':
            clfname = "Adaboost"
            from sklearn.ensemble import AdaBoostClassifier
            clf = AdaBoostClassifier(algorithm="SAMME.R")
            param_dist = {"n_estimators" : [10,20,30,40,50,75,100],
                          "learning_rate" : [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,0.9, 1.0]}


        # run randomized search
        n_iter_search = 50
        random_search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=n_iter_search, scoring=scorer, pre_dispatch=1)

        start = time.time()
        random_search.fit(df_x_shuf, df_y_shuf)
        print("RandomizedSearchCV took %.2f seconds for %d candidates"
              " parameter settings." % ((time.time() - start), n_iter_search))
        report(random_search.grid_scores_, features, 3, clfname)

print "Done."
