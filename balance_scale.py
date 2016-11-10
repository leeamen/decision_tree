#!/usr/bin/python
#coding:utf-8


import logging
import mylog
import mydecision_tree as ml
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

def MissEvaluate(x, y):
  param = {}
  param['adaboost'] = 10
  param['measure'] = 'info_gain'
  param['cv_fold'] = 10
  param['bootstrap'] = 10
  param['class_num'] = 4

  train_y = y
  for rate in [0.1, 0.3, 0.5, 0.7, 0.9]:
    train_x = ml.GenMissValueArray(x, rate, 625, 4)
    model = ml.Train(train_x, y, param)
    pred = model.Predict(train_x)
    logger.info('训练集测试准确率:%f', 1.0*sum(pred == train_y)/len(train_y))
    ml.HoldoutMethod(train_x, train_y, param)
    ml.CrossValidation(train_x, train_y, param)
    ml.Bootstrap(train_x, train_y, param)
    ml.Adaboost(train_x, train_y, param)
    logger.debug('-------------------')

if __name__ == '__main__':
#  filename = sys.argv[1]
  train_data = np.loadtxt('./data/balance-scale.data', delimiter = ',',# dtype = np.int, 
            converters={0:Str2Num})
  np.random.shuffle(train_data)
  logger.debug(train_data)
  
  train_x = train_data[:,1:]
  train_y = train_data[:,0]

  param = {}
  param['adaboost'] = 10
  param['measure'] = 'info_gain'
  param['cv_fold'] = 10
  param['bootstrap'] = 10
  param['class_num'] = 3
#  param['pre_pruning'] = 0.3
#  logger.info('使用prepruning,阈值:%f', param['pre_pruning'])
  #pessimistic pruning
#  param['pos_pruning'] = 'pessimistic'
#  param['factor'] = 1
#  logger.info('使用悲观剪枝法,惩罚值:%f', param['factor'])

#  x = ml.GenMissValueArray(train_x, 0.2, 625, 4)
#    logger.debug(train_x)
#    logger.debug(np.sum(np.isnan(x)))

#  MissEvaluate(train_x, train_y)
#  exit()
  model = ml.Train(train_x, train_y, param)
  pred = model.Predict(train_x)
#    logger.debug('tree:%s', model.root)
#    logger.debug('pred:%s', pred)
#    logger.debug('   y:%s', train_y)
  logger.info('训练集测试准确率:%f', 1.0*sum(pred == train_y)/len(train_y))
  ml.HoldoutMethod(train_x, train_y, param)
  ml.CrossValidation(train_x, train_y, param)
  ml.Bootstrap(train_x, train_y, param)
  ml.Adaboost(train_x, train_y, param)
  logger.debug('-------------------')
