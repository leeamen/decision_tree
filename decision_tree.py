#!/usr/bin/python
#coding:utf-8

import mylog
import myequation as eq
import logging
import numpy as np
import os
import sys
import pprint as pp
import copy
import random

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class MyTreeNode(object):
  def __init__(self, x, y, x_rows, features, param):
    self.father = None
    self.children = {}
    self.split_attr = -1

    self.x = x
    self.y = y
    self.x_rows = x_rows;
    self.features = features[:]
    self.param = copy.deepcopy(param)

    #分类
    self.label = -1

    #不纯性度量
    self.entropy = 0
    self.gini = 0
    self.class_error = 0
    self.info_gain = 0
    self.gain_ratio = 0

  def __str__(self):
    return self.Recursion(self)
    
  def Recursion(self, node):
    if len(node.children) <= 0:
      return '{叶子}'

    v = '{'+ str(node.split_attr)
    for key in node.children.keys():
      v += self.Recursion(node.children[key])
    v += '}'
    return v

  def HasChildren(self):
    return len(self.children) >= 1
  def GetChildren(self):
    alist = []
    for key in self.children.keys():
      alist.append((key,self.children[key]))
    return alist
  def SetEntropy(self, e):
    self.entropy = e
  def SetGini(self, g):
    self.gini = g
  def SetClassError(self, c):
    self.class_error = c
  def SetInfoGain(self, i):
    self.info_gain = i
  def SetGainRatio(self, g):
    self.gain_ratio = g
  
  def SetLabel(self, label):
    self.label = label
  def GetLabel(self):
    return self.label

  def GetSplitAttr(self):
    return self.split_attr
  def SetSplitAttr(self, split):
    self.split_attr = split

  def Addchild(self, attr_value, node):
    self.children[attr_value] = node
    node.AddFather(self)
  def AddFather(self,node):
    self.father = node

  def CalculateMeasure(self):
    num_class0 = sum(self.y[self.x_rows] == 0)
    num_class1 = sum(self.y[self.x_rows] == 1)

    if self.param['measure'] == 'gini':
      self.gini = eq.Gini([num_class0,num_class1])
    elif self.param['measure'] == 'entropy':
      self.entropy = eq.Entropy([num_class0,num_class1])
    elif self.param['measure'] == 'class_error':
      self.class_error = eq.ClassError([num_class0,num_class1])
    elif self.param['measure'] == 'info_gain':
      self.entropy = eq.Entropy([num_class0,num_class1])
    else:
      logger.error('error,参数 measure 错误:%s', self.param['measure'])

  def MeasureFeature(self, feature):
    f_dict = {}

#    logger.debug('self.x_rows:%s', self.x_rows)
#    logger.debug('计算划分的属性:%d', feature)
    for row in self.x_rows:
#      logger.debug('坐标的值%s:[%s]', (row, feature), str(self.x[row, feature]))
      if f_dict.has_key(self.x[row, feature]) == False:
        f_dict[self.x[row, feature]] = {}
        f_dict[self.x[row, feature]][self.y[row]] = 1
      elif f_dict[self.x[row, feature]].has_key(self.y[row]) == False:
        f_dict[self.x[row, feature]][self.y[row]] = 1
      else:
        f_dict[self.x[row, feature]][self.y[row]] += 1

    logger.debug(f_dict);
#    logger.debug('f_dict.keys() %s', f_dict.keys())

    #计算
    N = len(self.x_rows)
    he = 0.0
    for k in f_dict.keys():
     # logger.debug('%d', k)
     # logger.debug('%s', f_dict.keys())
     # logger.debug('%s', f_dict[k])
      nk = 0.0
      if not f_dict[k].has_key(0):
        f_dict[k][0] = 0
      if not f_dict[k].has_key(1):
        f_dict[k][1] = 0

      nk += f_dict[k][0] + f_dict[k][1]
      vk = self.param[self.param['measure']]([f_dict[k][0], f_dict[k][1]])
      he += float(nk) / N * vk
    if self.param['measure'] == 'info_gain':
      he = self.father.entropy - he
    #elif self.param['measure'] == 'gain_ratio':

    #logger.debug('不纯性度量值:%f', he)
    return (he, f_dict.keys())

  def FindBestSplit(self):
    measure = 100
    feature = -1
    feature_values = None

    choice = ''
    param_measure = self.param['measure']
    if param_measure == 'info_gain' or param_measure == 'gain_ratio':
      choice += 'max'
    else:
      choice += 'min'

#    logger.debug('self.features:%s', self.features);
    for f in self.features:
      (m, values) = self.MeasureFeature(f)
      if choice == 'min' and measure > m:
        measure = m
        feature = f
        feature_values = values
      elif choice == 'max' and measure < m:
        measure = m
        feature = f
        feature_values = values

    assert(feature >= 0)
    assert(not feature_values == None)
    return (feature , feature_values)

class MyModel(object):
  def __init__(self, param):
    self.param  = copy.deepcopy(param)
    self.root = None

  def StoppingCond(self, x, y, x_rows, features):
    #没有属性了
    if len(features) <= 0:
      return True
    #标签都是一类
    if sum(y == 0) == 0 or sum(y == 1) == 0:
      logger.debug('标签都是一类,结束,%s', y)
      return True
    #属性值都相同
    tmp_row = list(x[x_rows[0],features])
    for index in x_rows:
      if not tmp_row == list(x[index, features]):
        return False
    return True

  def Classify(self, y, x_rows):
    if sum(y[x_rows] == 0) > sum(y[x_rows] == 1):
      return 0
    return 1

#  def FindBestSplit(self, node):
#    return node.FindBestSplit()
      
    
#  def CalculateMeasure(self, node):
#    node.CalculateMeasure()
#
  def TreeGrowth(self,x, y, x_rows, features):
    r = self.StoppingCond(x, y, x_rows, features)
    if r == True:
      leaf = MyTreeNode(x, y, x_rows, features, self.param)
      leaf.SetLabel(self.Classify(y, x_rows))
      logger.debug('叶子节点,x_rows:%s,features:%s', x_rows, features)
      logger.debug('划分结束,label:%d', leaf.GetLabel())
      return leaf
    #还要继续划分
    root = MyTreeNode(x, y, x_rows, features, self.param)
    root.CalculateMeasure()

    (feature_index,feature_values) = root.FindBestSplit()
    root.SetSplitAttr(feature_index)
    #继续分裂
    new_features = features[:]
    new_features.remove(feature_index)
#    logger.debug('找到划分属性:%d,属性值如下:%s', feature_index, feature_values)
    for v in feature_values:
      new_x_row = np.intersect1d(x_rows, np.where(x[:,feature_index] == v)[0])
      logger.debug('孩子节点的x_rows:%s', new_x_row)
      child = self.TreeGrowth(x, y, new_x_row, new_features)
      root.Addchild(v, child)
    return root

  def fit(self, x, y, x_rows, features):
    self.root = self.TreeGrowth(x, y, x_rows, features)

  def Predict(self, x):
    y = np.array(len(x) * [-1])
    x_rows = []
    for i in range(0, len(x)):
      x_rows.append(i)

    self.RecursionTree(self.root, x, x_rows, y)
    return y

  def RecursionTree(self, node, x, x_rows, y):
    #叶子节点
    if not node.HasChildren():
      y[x_rows] = node.GetLabel()
      return

    feature = node.GetSplitAttr()
    for (value, child) in node.GetChildren():
      new_x_row = np.intersect1d(x_rows, np.where(x[:,feature] == value)[0])
      self.RecursionTree(child, x, new_x_row, y)

def Train(x,y,param = {}):
  #设置参数
  try:
    if param['measure'] == 'entropy':
      param[param['measure']] = eq.Entropy
    elif param['measure'] == 'gini':
      param[param['measure']] = eq.Gini
    elif param['measure'] == 'class_error':
      param[param['measure']] = eq.ClassError
    else:
      param[param['measure']] = eq.Entropy
  except:
    logger.error('error, 参数错误 measure')
    exit()
  #特征
  features = []
  try:
    for i in range(0, x.shape[1]):
      features.append(i)
  except:
    features = [0]
  #x_rows
  x_rows = []
  for i in range(0,len(x)):
    x_rows.append(i)

  #训练
  model = MyModel(param)
  model.fit(x, y, x_rows, features)

  return model

def RepeatRanddom(
def Bootstrap(x, y, param):


def CrossValidation(x, y, param):
  k = param['cv_fold']
  n = len(x) / k

  error_rate = 0.0
  for i in range(0, k):
    train_x = x[i*n :(i+1)*n, :]
    test_x = np.vstack((x[0:i*n, :], x[(i+1)*n:, :]))
    train_y = y[i*n : (i+1)*n]
    test_y = np.hstack((y[0: i*n], y[(i+1)*n:]))
    model = Train(train_x, train_y, param)
    prey = model.Predict(test_x)
    error_rate += 1.0 * sum(prey != test_y)/len(test_y)
#    logger.debug('%f', 1.0 * sum(prey != test_y)/len(test_y))
  logger.debug('错误率:%f', error_rate)

def HoldoutMethod(x, y, param):
  split = int(len(x) * 2.0/3)
  x1 = x[0:split, :]
  y1 = y[0:split]

  x2 = x[split:,:]
  y2 = y[split:]

  model = Train(x1, y1, param)
  prey = model.Predict(x2)
  logger.debug('准确率:%f', sum(prey == y2)/float(len(y2)))
  return model

if __name__ == '__main__':
  train = []
#  train.append([1,1,1,0])
#  train.append([0,2,2,0])
#  train.append([0,1,1,0])
#  train.append([1,2,2,0])
#  train.append([0,3,3,1])
#  train.append([0,2,2,0])
#  train.append([1,3,3,0])
#  train.append([0,1,1,1])
#  train.append([0,2,2,0])
#  train.append([0,1,1,1])

  train.append([1,1,0])
  train.append([0,2,0])
  train.append([0,1,0])
  train.append([1,2,0])
  train.append([0,3,1])
  train.append([0,2,0])
  train.append([1,3,0])
  train.append([0,1,1])
  train.append([0,2,0])
  train.append([0,1,1])
  train.append([1,1,0])
  train.append([0,2,0])
  train.append([0,1,0])
  train.append([1,2,0])
  train.append([0,3,1])
  train.append([0,2,0])
  train.append([1,3,0])
  train.append([0,1,1])
  train.append([0,2,0])
  train.append([0,1,1])

  train_data = np.array(train)

  train_x = train_data[:,0:-1]
  train_y = train_data[:,-1]

  param = {}
  param['measure'] = 'gini'
  param['cv_fold'] = 10
  model = Train(train_x, train_y, param)
  prey = model.Predict(train_x)
  logger.debug(model.root)
  logger.debug(prey)
  
  model = HoldoutMethod(train_x, train_y, param)
  CrossValidation(train_x, train_y, param)





