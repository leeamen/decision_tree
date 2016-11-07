#!/usr/bin/python
#coding:utf-8


import logging
import mylog
import decision_tree as ml
import sys
import os
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def Str2Num(s):
  if s == 'B':
    return 0
  elif s == 'R':
    return 2
  elif s == 'L':
    return 1
  else:
    return -1

if __name__ == '__main__':
#  filename = sys.argv[1]
  train_data = np.loadtxt('./data/balance-scale.data', delimiter = ',', dtype = np.int, converters={0:Str2Num})
  np.random.shuffle(train_data)
  logger.debug(train_data)
  
  train_x = train_data[:,1:]
  train_y = train_data[:,0]
  train_x = np.vstack((train_x, train_x))
  train_y = np.hstack((train_y, train_y))

  param = {}
  param['adaboost'] = 10
  param['measure'] = 'info_gain'
  param['cv_fold'] = 10
  param['bootstrap'] = 10
  param['class_num'] = 3
#  param['pre_pruning'] = 0.01
  #pessimistic pruning
#  param['pos_pruning'] = 'pessimistic'
#  param['factor'] = 0.5

  model = ml.Train(train_x, train_y, param)
  pred = model.Predict(train_x)
  logger.debug('准确率:%f', 1.0*sum(pred == train_y)/len(train_y))

#  ml.HoldoutMethod(train_x, train_y, param)
#  ml.CrossValidation(train_x, train_y, param)
#  ml.Bootstrap(train_x, train_y, param)
#  ml.Adaboost(train_x, train_y, param)
#
