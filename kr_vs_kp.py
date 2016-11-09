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

def str2num(s):
  a =  ['nowin', 'won'] + [chr(i) for i in range(97, 123)] 
  for i in range(0, len(a)):
    if s == a[i]:
      return float(i)
def MissEvaluate(x, y):
  param = {}
  param['adaboost'] = 10
  param['measure'] = 'info_gain'
  param['cv_fold'] = 10
  param['bootstrap'] = 10
  param['class_num'] = 4

  train_y = y
  for rate in [0.1, 0.3, 0.5, 0.7, 0.9]:
    train_x = ml.GenMissValueArray(x, rate, 3196, 36)
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
  train_data = np.loadtxt('./data/kr-vs-kp.data', delimiter = ',',
                          converters={i:str2num for i in range(0,37)})
  np.random.shuffle(train_data)
  logger.debug(train_data)
  
  train_x = train_data[:,0:-1]
  train_y = train_data[:,-1]

  param = {}
  param['adaboost'] = 10
  param['measure'] = 'info_gain'
  param['cv_fold'] = 10
  param['bootstrap'] = 10
  param['class_num'] = 2
#  param['pre_pruning'] = 0.3
#  logger.info('使用prepruning,阈值:%f', param['pre_pruning'])
  #pessimistic pruning
#  param['pos_pruning'] = 'pessimistic'
#  param['factor'] = 1
#  logger.info('使用悲观剪枝法,惩罚值:%f', param['factor'])

#  logger.debug(train_x)
#  logger.debug(train_y)
#  MissEvaluate(train_x, train_y)
#  train_x = ml.GenMissValueArray(train_x, 0.2, 3196, 36)
#    logger.debug(train_x)
#    logger.debug(np.sum(np.isnan(x)))
  #  param['pre_pruning'] = 0.3
  #  logger.info('使用prepruning,阈值:%f', param['pre_pruning'])
    #pessimistic pruning
  #  param['pos_pruning'] = 'pessimistic'
  #  param['factor'] = 1
  #  logger.info('使用悲观剪枝法,惩罚值:%f', param['factor'])

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

