#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import Orange

data =  Orange.data.Table("credit_data_orange.csv")
cn2_leaner = Orange.classification.rules.CN2Learner()
data_test, data_traning = Orange.evaluation.testing.sample(data, n = 0.3)
clf = cn2_leaner(data_test)

result = Orange.evaluation.testing.TestOnTestData(data_traning, data_test, [lambda x: clf])
print(Orange.evaluation.CA(result))