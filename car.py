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

def str2num(s):
  a = ['unacc', 'acc', 'good', 'vgood','vhigh', 'high', 'med',
        'low', 'small', 'med', 'big', '2', '3', '4', '5more','more']
  for i in range(0, len(a)):
    if s == a[i]:
      return i

if __name__ == '__main__':
#  filename = sys.argv[1]
  train_data = np.loadtxt('./data/car.data', delimiter = ',', dtype = np.int,
          converters={i:str2num for i in range(0,7)})
  np.random.shuffle(train_data)
  logger.debug(train_data)
  
  train_x = train_data[:,0:-1]
  train_y = train_data[:,-1]
#  train_x = np.vstack((train_x, train_x))
#  train_y = np.hstack((train_y, train_y))

  param = {}
  param['adaboost'] = 10
  param['measure'] = 'info_gain'
  param['cv_fold'] = 10
  param['bootstrap'] = 10
  param['class_num'] = 4
#  param['pre_pruning'] = 0.5
  #pessimistic pruning
  param['pos_pruning'] = 'pessimistic'
  param['factor'] = 1.0

#  model = ml.Train(train_x, train_y, param)
#  pred = model.Predict(train_x)
#  logger.debug('准确率:%f', 1.0*sum(pred == train_y)/len(train_y))
  ml.HoldoutMethod(train_x, train_y, param)
  ml.CrossValidation(train_x, train_y, param)
  ml.Bootstrap(train_x, train_y, param)
  ml.Adaboost(train_x, train_y, param)
#
