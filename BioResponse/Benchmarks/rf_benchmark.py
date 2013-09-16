#!/usr/bin/env python

from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
import evalfun
import math
import numpy as np

def main():
#    train = csv_io.read_data("../Data/train.csv")
    dataset = np.genfromtxt(open('../Data/train.csv','r'), delimiter=',', dtype='f8')[1:]
    target = np.array([x[0] for x in dataset])
    train = np.array([x[1:] for x in dataset])
#    test = csv_io.read_data("../Data/test.csv")

    rf = RandomForestClassifier(n_estimators=100)

    #Simple K-Fold cross validation. 5 folds.
    cv = cross_validation.KFold(len(train), k=5, indices=False)

    #iterate through the training and test cross validation segments and
    #run the classifier on each one, aggregating the results into a list
    results = []
    for traincv, testcv in cv:
        rf.fit(train[traincv], target[traincv])
        probas = rf.predict_proba(train[testcv])
        results.append(evalfun.llfun(target[testcv], [x[1] for x in probas]) )

    #print out the mean of the cross-validated results
    print "Results: " + str(np.array(results).mean() )

#    rf.fit(train, target)
#    predicted_probs = rf.predict_proba(test)
#    predicted_probs = ["%f" % x[1] for x in predicted_probs]
#    csv_io.write_delimited_file("../Submissions/rf_benchmark.csv",
#                                predicted_probs)

if __name__=="__main__":
    main()
