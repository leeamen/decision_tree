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
    self.error_num = 0

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
  def GetXLen(self):
    return len(self.x)
  def GetY(self):
    return self.y
  def GetXRows(self):
    return self.x_rows
  def GetEntropy(self):
    return self.entropy
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
    #node.SetFather(self)
  def ClearChildren(self):
    self.children.clear()
  def SetFather(self,node):
    self.father = node

  def HasAncestor(self, node):
    p = self.father
    while not p == None:
      if p == node:
        return True
      p = p.father
    return False
  def CalcuErrorNum(self):
    self.error_num = sum(self.label != self.y[self.x_rows])

  def GetErrorNum(self):
    return self.error_num
  def CalcuMeasure(self):
#    num_class0 = sum(self.y[self.x_rows] == 0)
#    num_class1 = sum(self.y[self.x_rows] == 1)
    class_num = self.param['class_num']
    vector = [sum(self.y[self.x_rows] == i) for i in range(0, class_num)]

    if self.param['measure'] == 'gini':
      self.gini = eq.Gini(vector)
    elif self.param['measure'] == 'entropy':
      self.entropy = eq.Entropy()
    elif self.param['measure'] == 'class_error':
      self.class_error = eq.ClassError(vector)
    elif self.param['measure'] == 'info_gain':
      self.entropy = eq.Entropy(vector)
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
    #计算
    N = len(self.x_rows)
    he = 0.0
    for k in f_dict.keys():
     # logger.debug('%d', k)
     # logger.debug('%s', f_dict.keys())
     # logger.debug('%s', f_dict[k])
      nk = 0.0
     # if not f_dict[k].has_key(0):
     #   f_dict[k][0] = 0
     # if not f_dict[k].has_key(1):
     #   f_dict[k][1] = 0

      class_num = self.param['class_num']
      for i in range(0, class_num):
        try:
          nk += f_dict[k][i]
        except:
          f_dict[k][i] = 0
          nk += 0

      vk = self.param[self.param['measure']]([f_dict[k][i] for i in range(0,class_num)])
      he += float(nk) / N * vk
    if self.param['measure'] == 'info_gain':
      if not self.father == None:
        he = self.father.GetEntropy() - he
    #elif self.param['measure'] == 'gain_ratio':

    #logger.debug('不纯性度量值:%f', he)
    return (he, f_dict.keys())

  def FindBestSplit(self):
    feature = -1
    feature_values = None
    choice = ''
    param_measure = self.param['measure']
    if param_measure == 'info_gain' or param_measure == 'gain_ratio':
      choice += 'max'
      measure = -1
    else:
      choice += 'min'
      measure = 100

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

#    logger.debug('feature:%d', feature)
    assert(feature >= 0)
    assert(not feature_values == None)
    return (measure, feature , feature_values)
  def CalcuErrorNumForLabel(self, label):
    return sum(self.y[self.x_rows] == label)

class MyModel(object):
  def __init__(self, param):
    self.param  = copy.deepcopy(param)
    self.root = None
    self.leafs = []

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
    class_num = self.param['class_num']

    max_num_label = sum(y[x_rows] == 0)
    max_label = 0
    for i in range(1, class_num):
      num_label_i = sum(y[x_rows] == i)
      if num_label_i > max_num_label:
        max_label = i
        max_num_label = num_label_i
    return max_label

#  def FindBestSplit(self, node):
#    return node.FindBestSplit()
      
    
#  def CalcuMeasure(self, node):
#    node.CalcuMeasure()
#
  def GetLeafNumNotAncestor(self, node):
    total = len(self.leafs)
    for leaf in self.leafs:
      if leaf.HasAncestor(node) == True:
        total -= 1
    return total
  def CalcuNodePEP(self, node):
    N = node.GetXLen()
    label = self.Classify(node.GetY(), node.GetXRows())
#    logger.debug('标签:%d', label)
    #计算错误个数
    error_num = node.CalcuErrorNumForLabel(label)
    error_num += self.GetErrorNumNotAncestor(node)

    omega = self.GetOmega()
    leaf_num = self.GetLeafNumNotAncestor(node)
    #加上自己
    leaf_num += 1
    #计算PEP
    return (error_num + omega * leaf_num) / N

  def GetErrorNumNotAncestor(self, node):
    num = 0
    for leaf in self.leafs:
      if leaf.HasAncestor(node) == False:
        num+=leaf.GetErrorNum()
    return num
  def Pessimistic(self, node):
    if len(node.GetChildren()) <= 0:
      return True

    pep = self.CalcuNodePEP(node)
    tree_pep = self.CalcuTreePEP()
#    logger.debug('tree_pep:%f,如果剪枝,剪后pep:%f', tree_pep, pep)
    if pep < tree_pep:
      #更新叶子节点列表
      self.CutLeafs(node)
      node.ClearChildren()
      node.SetLabel(self.Classify(node.GetY(), node.GetXRows))
      node.CalcuErrorNum()
      logger.debug('剪枝,tree_pep:%f,如果剪枝,剪后pep:%f', tree_pep, pep)
      return True
    #遍历
    for (v, child) in node.GetChildren():
      self.Pessimistic(child)

  def CutLeafs(self, node):
    for i in range(0, len(self.leafs)):
      if self.leafs[i].HasAncestor(node) == True:
        self.leafs.pop(i)
    self.leafs.append(node)

  def GetOmega(self):
    try:
      omega = self.param['factor']
    except:
      omega = 0.5 
    return omega

  def CalcuTreePEP(self):
    pep = 0.0
    leaf_num = len(self.leafs)
    N = self.root.GetXLen()
    error_num = 0
    for leaf in self.leafs:
      error_num += leaf.GetErrorNum()
    omega = self.GetOmega()
    return (error_num + omega * leaf_num)/N

  def PosPruning(self):
    if self.param['pos_pruning'] == 'pessimistic':
      self.Pessimistic(self.root)

  def PrePruning(self, measure):
    try:
      threshold = self.param['pre_pruning']
    except:
      return False
    return measure < threshold

  def TreeGrowth(self,father,x, y, x_rows, features):
    r = self.StoppingCond(x, y, x_rows, features)
    if r == True:
      leaf = MyTreeNode(x, y, x_rows, features, self.param)
      leaf.SetLabel(self.Classify(y, x_rows))
      leaf.CalcuErrorNum()
      #logger.debug('叶子节点,x_rows:%s,features:%s', x_rows, features)
      #logger.debug('划分结束,label:%d', leaf.GetLabel())
      self.leafs.append(leaf)
      return leaf
    #还要继续划分
    root = MyTreeNode(x, y, x_rows, features, self.param)
    root.CalcuMeasure()
    root.SetFather(father)

    (measure, feature_index,feature_values) = root.FindBestSplit()
    #先剪枝
    if self.PrePruning(measure) == True:
      logger.debug('prepruning,增益:%f,阈值:%s', measure, self.param['pre_pruning'])
      leaf = MyTreeNode(x, y, x_rows, features, self.param)
      leaf.SetLabel(self.Classify(y, x_rows))
      leaf.CalcuErrorNum()
      self.leafs.append(leaf)
      return leaf
    root.SetSplitAttr(feature_index)
    #继续分裂
    new_features = features[:]
    new_features.remove(feature_index)
#    logger.debug('找到划分属性:%d,属性值如下:%s', feature_index, feature_values)
    for v in feature_values:
      new_x_row = np.intersect1d(x_rows, np.where(x[:,feature_index] == v)[0])
      #logger.debug('孩子节点的x_rows:%s', new_x_row)
      child = self.TreeGrowth(root, x, y, new_x_row, new_features)
      root.Addchild(v, child)
    return root

  def fit(self, x, y, x_rows, features):
    self.root = self.TreeGrowth(None, x, y, x_rows, features)

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

  #后剪枝
  if param.has_key('pos_pruning'):
    model.PosPruning()

  return model

def RepeatRandom(start, end, N):
  rand_list = []
  for i in range(0,N):
    rand_list.append(random.randint(start, end))
  return rand_list

def GetRepeatSample(x, y):
  rands = RepeatRandom(0, len(x) - 1, len(x))
  dimension = 0
  try:
    dimension = x.shape[1]
  except:
    logger.warn('error,x是一维数组:%s', x)
    dimension = 1
  if dimension > 1:
    rand_x = x[rands[0], :]
  else:
    rand_x = x[rands[0]]

  rand_y = [y[rands[0]]]
  for i in range(1, len(rands)):
    rand_y.append(y[rands[i]])
    if dimension > 1:
      rand_x = np.vstack((rand_x, x[rands[i], :]))
    else:
      rand_x = np.append(rand_x, rands[i])
  return (rand_x, np.array(rand_y, dtype = np.int))

def Bootstrap(x, y, param):
  accurate = 0.0
  b = param['bootstrap']
  for i in range(0, b):
    (train_x, train_y) = GetRepeatSample(x, y)
#    logger.debug('train_x:%s,train_y:%s', train_x, train_y)
    #train
    model = Train(train_x, train_y, param)
    pred_r = model.Predict(train_x)
    pred_s = model.Predict(x)

    acc_r  = 1.0*sum(pred_r == train_y) / len(train_y)
    acc_s = 1.0 * sum(pred_s == y) / len(x)
    accurate += 0.632 * acc_r + 0.368 * acc_s
  accurate /= b
  logger.debug('bootstrap 准确率:%f', accurate)

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
    pred = model.Predict(test_x)
    error_rate += 1.0 * sum(pred != test_y)/len(test_y)
#    logger.debug('%f', 1.0 * sum(pred != test_y)/len(test_y))
  logger.debug('交叉验证平均错误率:%f', error_rate / k)

def Adaboost(x, y, param):
  #初始化权重
  N = len(x)
  k = param['adaboost']

  weight = np.array([1.0/N] * N, dtype = np.float)
  alpha = np.zeros(k, dtype = np.float)
  Z = np.zeros(k, dtype = np.float)
  classifiers = []
  for i in range(0, k):
    (train_x, train_y) = GetRepeatSample(x, y)
    classifer = Train(train_x, train_y, param)
    classifiers.append(classifer)
    pred = classifer.Predict(train_x)
    epsilon = 1.0/N * np.dot(weight, np.array(pred == train_y, dtype = np.float))
    if epsilon > 0.5:
      weight = np.array([1.0/N] * N, dtype = np.float)
    #更新权值
    alpha[i] = 1.0/2 * np.log((1-epsilon)/epsilon)
    tmp = np.exp(alpha[i]) * (pred != train_y) + np.exp(-1.0 * alpha[i])*(pred == train_y)
#    logger.debug('y   :%s', train_y)
#    logger.debug('pred:%s', pred)
#    logger.debug('tmp:%s', tmp)
    Z[i] = np.sum(tmp * weight)
    weight = weight / Z[i] * tmp
#    logger.debug('weight:%s', weight)
#  logger.debug('alpha:%s', alpha)
#  logger.debug('Z:%s', Z)
  #计算最终标签
  preds = {}
  for i in range(0, len(classifiers)):
    preds[i] = classifiers[i].Predict(x)
    logger.debug('pred:%s', preds[i])
    logger.debug('y   :%s', y)

  args = np.empty((0, N), dtype = np.float)
  for i in range(0, param['class_num']):
    arg = np.zeros(N, dtype = np.float)
    for j in range(0, len(classifiers)):
      arg += alpha[j] * (preds[j] == i)
    args = np.vstack((args, arg))

  final_y = Argmax(args, N)
  logger.debug('pred:%s', final_y)
  logger.debug('y   :%s', y)
  acc = 1.0*sum(final_y == y) / len(y)
  logger.debug('Adaboost 准确率为:%f', acc)

#求每一列最大的值所在的行号,即标签，多分类adaboost
def Argmax(x, N):
  y = np.zeros(N, dtype = np.int)
  for i in range(0, N):
    y[i] = np.where(x[:,i] == np.max(x[:,i]))[0]
  return y

def HoldoutMethod(x, y, param):
  split = int(len(x) * 2.0/3)
  x1 = x[0:split, :]
  y1 = y[0:split]

  x2 = x[split:,:]
  y2 = y[split:]

  model = Train(x1, y1, param)
  pred = model.Predict(x2)
  logger.debug('holdout method准确率:%f', 1.0*sum(pred == y2)/len(y2))

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

  train.append([1,1,2])
  train.append([0,2,2])
  train.append([0,1,0])
  train.append([1,2,0])
  train.append([0,3,1])
  train.append([0,2,0])
  train.append([1,3,2])
  train.append([0,1,1])
  train.append([0,2,0])
  train.append([0,1,1])
  train.append([1,1,2])
  train.append([0,2,0])
  train.append([0,1,2])
  train.append([1,2,0])
  train.append([0,3,1])
  train.append([0,2,0])
  train.append([1,3,0])
  train.append([0,1,1])
  train.append([0,2,0])
  train.append([0,1,2])

  train_data = np.array(train)

  train_x = train_data[:,0:-1]
  train_y = train_data[:,-1]

  param = {}
  param['adaboost'] = 10
  param['measure'] = 'info_gain'
  param['cv_fold'] = 10
  param['bootstrap'] = 10
  param['class_num'] = 3
#  param['pre_pruning'] = 0.01
  #pessimistic pruning
  param['pos_pruning'] = 'pessimistic'
  param['factor'] = 0.5

  model = Train(train_x, train_y, param)
  pred = model.Predict(train_x)
  logger.debug('准确率:%f', 1.0*sum(pred == train_y)/len(train_y))
  
  HoldoutMethod(train_x, train_y, param)
  CrossValidation(train_x, train_y, param)
  Bootstrap(train_x, train_y, param)
  Adaboost(train_x, train_y, param)

