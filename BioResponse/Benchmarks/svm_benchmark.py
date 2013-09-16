#!/usr/bin/env python

from sklearn import svm
from sklearn import cross_validation
import evalfun
import numpy as np

def main():
#    train = csv_io.read_data("../Data/train.csv")
#    target = [x[0] for x in train]
#    train = [x[1:] for x in train]
#    test = csv_io.read_data("../Data/test.csv")

    dataset = np.genfromtxt(open('../Data/train.csv','r'), delimiter=',', dtype='f8')[1:]
    target = np.array([x[0] for x in dataset])
    train = np.array([x[1:] for x in dataset])

    cfr = svm.SVC(probability=True)

    #Simple K-Fold cross validation. 5 folds.
    cv = cross_validation.KFold(len(train), k=5, indices=False)

    results = []
    for traincv, testcv in cv:
        cfr.fit(train[traincv], target[traincv])
        probas = cfr.predict_proba(train[testcv])
        results.append(evalfun.llfun(target[testcv], [x[1] for x in probas]) )

    #print out the mean of the cross-validated results
    print "Results: " + str(np.array(results).mean() )

#    svc.fit(train, target)
#    predicted_probs = svc.predict_proba(test)
#    predicted_probs = ["%f" % x[1] for x in predicted_probs]
#    csv_io.write_delimited_file("../Submissions/svm_benchmark.csv",
#                                predicted_probs)

if __name__=="__main__":
    main()
