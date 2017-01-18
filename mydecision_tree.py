#!/usr/bin/python
#coding:utf-8

from mltoolkits import *
import myequation as eq
import logging
import numpy as np
import os
import sys
import pprint as pp
import copy
import random
import numpy.lib.arraysetops as arraysetops
import time
random.seed(time.time())

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class MyTreeNode(object):
  def __init__(self, x, y, x_rows, features, param):
    self.father = None
    self.children = {}
    self.split_attr = -1
    self.error_num = 0

    self.x = np.array(x, dtype = x.dtype)
    self.y = np.array(y, dtype = y.dtype)
    self.x_rows = x_rows;
    self.features = copy.deepcopy(features)
    self.param = copy.deepcopy(param)

    #分类
    self.label = -1

    #不纯性度量
    self.entropy = -1
    self.gini = -1
    self.class_error = -1
    self.info_gain = -1
    self.gain_ratio = -1

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
    node.SetFather(self)
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
#    logger.debug('叶子节点,错误数量:%d', self.error_num)

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
      self.entropy = eq.Entropy(vector)
    elif self.param['measure'] == 'class_error':
      self.class_error = eq.ClassError(vector)
    elif self.param['measure'] == 'info_gain':
      self.entropy = eq.Entropy(vector)
    else:
      logger.error('error,参数 measure 错误:%s', self.param['measure'])

  def ProcMissValue(self, feature):
    #缺失值处理
    stat_dict = {}
    miss_rows = []
    for row in self.x_rows:
      if stat_dict.has_key(self.x[row, feature]) == False:
        if np.isnan(self.x[row, feature]) == True:
#          logger.debug('发现nan,row:%d', row)
          miss_rows.append(row)
        else:
          stat_dict[self.x[row, feature]] = 1
      else:
        stat_dict[self.x[row, feature]] += 1
    #most frequent value
    most_frequent = None
    for key in stat_dict.keys():
      if most_frequent == None:
        most_frequent = key
      elif stat_dict[most_frequent] < stat_dict[key]:
        most_frequent = key
#    logger.debug('most_frequent float:%f', most_frequent)
#    logger.debug('most_frequent:%s', str(most_frequent))
#    logger.debug('miss_rows:%s', miss_rows)
    if most_frequent == None:# or len(miss_rows) == 0:
      logger.debug('都是nan,随机生成')
#      max_value = np.max(self.x[:,feature])
#      min_value = np.min(self.x[:,feature])
#      for row in miss_rows:
#        self.x[row, feature] = float(random.randint(int(min_value), int(max_value)))
      return False
      
    for row in miss_rows:
#      logger.debug('self.x[row,feature]:%s', str(self.x[row,feature]))
      self.x[row,feature] = most_frequent
    return True

  def MeasureFeature(self, feature):
    f_dict = {}

#    logger.debug('self.x_rows:%s', self.x_rows)
#    logger.debug('计算划分的属性:%d', feature)
    miss_rows = []
    for row in self.x_rows:
#      logger.debug('坐标的值%s:[%s]', (row, feature), str(self.x[row, feature]))
      if f_dict.has_key(self.x[row, feature]) == False:
        f_dict[self.x[row, feature]] = {}
        try:
          f_dict[self.x[row, feature]][self.y[row]] = 1
        except KeyError:
#          logger.debug('%s,%s', str(self.x[row, feature]), str(self.y[row]))
#          logger.debug('nan')
          #处理缺失值
          r = self.ProcMissValue(feature)
          if r == True:
            return self.MeasureFeature(feature)
          #处理不了,可能全是缺失值,放弃处理nan,数据不进行划分
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
      if nk <= 0.0:
        continue
      vk = self.param[self.param['measure']]([f_dict[k][i] for i in range(0,class_num)])
      he += float(nk) / N * vk
#      logger.debug('%d, %f', nk, vk)
    final = he
    if self.param['measure'] == 'info_gain':
      final = self.entropy - final
    if final < 0.0:
      logger.warn('error,信息增益为:%f', final)
      logger.warn('error,father    :%f', self.entropy)
      logger.warn('error,划分的商  :%f', he)
#      logger.warn(vk)
#      logger.warn(nk)
    #elif self.param['measure'] == 'gain_ratio':

    #logger.debug('不纯性度量值:%f', he)
    return (final, f_dict.keys())

  def FindBestSplit(self):
    feature = -1
    feature_values = None
    choice = ''
    param_measure = self.param['measure']
    if param_measure == 'info_gain' or param_measure == 'gain_ratio':
      choice += 'max'
      measure = -100
    else:
      choice += 'min'
      measure = 100

#    logger.debug('self.features:%s', self.features);
#    logger.debug('self.x_rows:%s', self.x_rows);
#    logger.debug('y.x_rows:%s', self.y[self.x_rows]);
#    logger.debug('self.entroy:%f', self.entropy)
    for f in self.features:
      #全是缺失值，不划分此属性
      if sum(np.isnan(self.x[self.x_rows, f])) == len(self.x_rows):
        continue
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
#    logger.debug('measure:%f', measure)
    if not measure >= 0.0:
      return (-1, -1, [])
    assert(measure >= 0.0)
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
    #没有属性了,或者因为缺失值导致x_rows为空
    if len(features) <= 0 or len(x_rows) <= 0:
      if len(features) > 0:
        logger.debug('features:%s,x_rows:%s', features,x_rows)
      return True
    if np.max(y[x_rows]) == np.min(y[x_rows]):
     # logger.debug('标签都是一类,结束,%s', y[x_rows])
      return True
    #属性值都相同
    tmp_row = list(x[x_rows[0],features])
#    logger.debug('tmp_row:%s', tmp_row)
    for index in x_rows:
#      logger.debug('row:%s', list(x[index, features]))
      if not tmp_row == list(x[index, features]):
        return False
    return True

  def ClassifyRandom(self, class_num):
    return random.randint(0, class_num - 1)
  def Classify(self, y, x_rows):
    class_num = self.param['class_num']

    #由于缺失值导致x_rows是空的
    if len(x_rows) <= 0:
      logger.warn('不该到这里')
      return random.randint(0, class_num - 1)
#    max_num_label = sum(y[x_rows] == 0)
    max_num_label = None
    max_label = -1
    for i in range(0, class_num):
      num_label_i = sum(y[x_rows] == i)
      if max_num_label is None:
        max_num_label = num_label_i
        max_label = i
      elif num_label_i >= max_num_label:
        max_label = i
        max_num_label = num_label_i
    assert(max_label >= 0)
    assert(max_num_label >= 0)
    return max_label

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
#    logger.debug('错分类个数:%d', error_num)

    omega = self.GetOmega()
    leaf_num = self.GetLeafNumNotAncestor(node)
    #加上自己
    leaf_num += 1
    #计算PEP
#    logger.debug('计算如果剪枝后,leaf_num:%d', leaf_num)
#    logger.debug('计算如果剪枝后,error_num:%d', error_num)
    return (leaf_num,(error_num + omega * leaf_num) / (1.0*N))

  def GetErrorNumNotAncestor(self, node):
    num = 0
    for leaf in self.leafs:
      if leaf.HasAncestor(node) == False:
        num+=leaf.GetErrorNum()
    return num
  def Pessimistic(self, node):
    if len(node.GetChildren()) <= 0:
      return True

    (leaf_num,pep) = self.CalcuNodePEP(node)
    (leaf_num_tree,tree_pep) = self.CalcuTreePEP()
#    logger.debug('tree_pep:%f,如果剪枝,剪后pep:%f', tree_pep, pep)
    if pep < tree_pep:
      logger.debug('剪枝,tree_pep:%f,树叶子个数:%d,剪后pep:%f,叶子个数:%d', tree_pep, leaf_num_tree,pep,leaf_num)
      #更新叶子节点列表
      self.CutLeafs(node)
      node.ClearChildren()
#      logger.debug('合并节点x_rows:%s',node.GetXRows())
      node.SetLabel(self.Classify(node.GetY(), node.GetXRows()))
      node.CalcuErrorNum()
      return True
    #遍历
    for (v, child) in node.GetChildren():
      self.Pessimistic(child)

  def CutLeafs(self, node):
    logger.debug('剪枝之前leafs:%d', len(self.leafs))
#    logger.debug(self.leafs)
    #倒序删除
    cutsum = 0
    for i in range(len(self.leafs)-1, -1, -1):
      if self.leafs[i].HasAncestor(node) == True:
        self.leafs.pop(i)
        cutsum+=1
    self.leafs.append(node)
    logger.debug('减掉结点个数:%d',cutsum)

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

#    logger.debug('计算树,leaf_num:%d', leaf_num)
#    logger.debug('计算树,error_num:%d', error_num)
#    logger.debug('N:%d',N)
    return (leaf_num,(error_num + omega * leaf_num)/(1.0*N))

  def PosPruning(self):
    if self.param['pos_pruning'] == 'pessimistic':
      #logger.debug('进行后剪枝,pessimistic pruning.')
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
    #都是缺失值，无法划分
    if measure < 0.0:
#      logger.debug('全是缺失值,不进行划分,节点数据个数:%d!', len(x_rows))
#      root.SetLabel(self.ClassifyRandom(self.param['class_num']))
      root.SetLabel(self.Classify(y, x_rows))
#      logger.debug('设置标签:%d', root.GetLabel())
      root.CalcuErrorNum()
      self.leafs.append(root)
      return root
      
    #先剪枝
    if self.PrePruning(measure) == True:
#      logger.debug('prepruning,增益:%f,阈值:%s', measure, self.param['pre_pruning'])
      leaf = MyTreeNode(x, y, x_rows, features, self.param)
      leaf.SetLabel(self.Classify(y, x_rows))
      leaf.CalcuErrorNum()
      self.leafs.append(leaf)
      return leaf
    #logger.debug('FindBestSplit feature_index:%d,measure:%f', feature_index, measure)
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
    x_rows = [i for i in range(0, len(x))]

    self.RecursionTree(self.root, x, x_rows, y)
    return y

  def RecursionTree(self, node, x, x_rows, y):
    #没有节点
    if len(x_rows) <= 0:
      return
    #叶子节点
    if node.HasChildren() == False:
      y[x_rows] = node.GetLabel()
#      logger.debug('predict 获取标签:%d,x_rows:%s', node.GetLabel(),x_rows)
      return

    feature = node.GetSplitAttr()
    rest_x_row = np.array(x_rows, dtype = np.int)
    for (value, child) in node.GetChildren():
      new_x_row = np.intersect1d(x_rows, np.where(x[:,feature] == value)[0])
      rest_x_row = arraysetops.setxor1d(rest_x_row, new_x_row, True)
      self.RecursionTree(child, x, new_x_row, y)

    #训练不充分导致测试集有未知数据不能向下划分预测
    y[rest_x_row] = self.Classify(y, x_rows)

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
#  logger.debug('树:%s', model.root)
  return model

def GetRandom2DArray(rate, x_dimension, y_dimension):
  total = x_dimension * y_dimension
  num = int(rate * total)
  rand_arr = random.sample(xrange(0, total), num)
#  logger.debug(rand_arr)
  rand2d = np.empty((0,2), dtype = np.int)
  for i in range(0, len(rand_arr)):
    loc_x = rand_arr[i]/y_dimension
    loc_y = rand_arr[i]%y_dimension
    rand2d = np.vstack((rand2d, [loc_x, loc_y]))
  return rand2d
def GenMissValueArray(arr, rate, x_dimension, y_dimension):
  x = np.array(arr)
  rand_arr = GetRandom2DArray(rate, x_dimension, y_dimension)
#  logger.debug(rand_arr)
  for i in range(0, len(rand_arr)):
    x[rand_arr[i][0], rand_arr[i][1]] = np.nan
  return x
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
  logger.info('10次抽样Bootstrap准确率:%f', accurate)

def CrossValidation(x, y, param):
  k = param['cv_fold']
  n = len(x) / k

  error_rate = 0.0
  for i in range(0, k):
    train_x = x[i*n :(i+1)*n, :]
    test_x = np.vstack((x[0:i*n, :], x[(i+1)*n:, :]))
    train_y = y[i*n : (i+1)*n]
    test_y = np.hstack((y[0: i*n], y[(i+1)*n:]))

#    logger.debug('train_x:%s,test_x:%s,train_y:%s,test_y:%s',train_x.shape,test_x.shape,train_y.shape,test_y.shape)
    model = Train(train_x, train_y, param)
    pred = model.Predict(test_x)
    error_rate += 1.0 * sum(pred != test_y)/len(test_y)
#    logger.debug('%f', 1.0 * sum(pred != test_y)/len(test_y))
  logger.info('10折交叉验证平均准确率:%f', 1.0 - error_rate / k)

def Adaboost(x, y, param):
  #初始化权重
  N = len(x)
  k = param['adaboost']

  weight = np.array([1.0/N] * N, dtype = np.float)
  alpha = np.zeros(k, dtype = np.float)
  Z = np.zeros(k, dtype = np.float)
  classifiers = []
  i = 0
  while i < k:
    (train_x, train_y) = GetRepeatSample(x, y)
    classifer = Train(train_x, train_y, param)
    classifiers.append(classifer)
    pred = classifer.Predict(train_x)
    epsilon = 1.0/N * np.dot(weight, np.array(pred == train_y, dtype = np.float))
    if epsilon > 0.5:
      weight = np.array([1.0/N] * N, dtype = np.float)
      continue
    #更新权值
    alpha[i] = 1.0/2 * np.log((1-epsilon)/epsilon)
    tmp = np.exp(alpha[i]) * (pred != train_y) + np.exp(-1.0 * alpha[i])*(pred == train_y)
#    logger.debug('y   :%s', train_y)
#    logger.debug('pred:%s', pred)
#    logger.debug('tmp:%s', tmp)
    Z[i] = np.sum(tmp * weight)
    weight = weight / Z[i] * tmp
    i+=1
#    logger.debug('weight:%s', weight)
#  logger.debug('alpha:%s', alpha)
#  logger.debug('Z:%s', Z)
  #计算最终标签
  preds = {}
  for i in range(0, len(classifiers)):
    preds[i] = classifiers[i].Predict(x)
#    logger.debug('pred:%s', preds[i])
#    logger.debug('y   :%s', y)

  args = np.empty((0, N), dtype = np.float)
  for i in range(0, param['class_num']):
    arg = np.zeros(N, dtype = np.float)
    for j in range(0, len(classifiers)):
      arg += alpha[j] * (preds[j] == i)
    args = np.vstack((args, arg))

  final_y = Argmax(args, N)
#  logger.debug('pred:%s', final_y)
#  logger.debug('y   :%s', y)
  acc = 1.0*sum(final_y == y) / len(y)
  logger.info('Adaboost 准确率为:%f', acc)

#求每一列最大的值所在的行号,即标签,多分类adaboost,矩阵:class_num * N
def Argmax(x, N):
  y = np.zeros(N, dtype = np.int)
  for i in range(0, N):
    try:
      y[i] = np.where(x[:,i] == np.max(x[:,i]))[0]
    except:
#      logger.debug('%s', np.where(x[:,i] == np.max(x[:,i]))[0])
#      logger.debug('%s', x[:,i])
      y[i] = np.where(x[:,i] == np.max(x[:,i]))[0][0]
  return y

def HoldoutMethod(x, y, param):
  split = int(len(x) * 2.0/3)
  x1 = x[0:split, :]
  y1 = y[0:split]

  x2 = x[split:,:]
  y2 = y[split:]

#  logger.debug('%s,%s,%s,%s',x1.shape, y1.shape, x2.shape, y2.shape)
  model = Train(x1, y1, param)
  pred = model.Predict(x2)
#  logger.debug('%s,%s,%s,%s',x1.shape, y1.shape, x2.shape, y2.shape)
#  logger.debug('x1:%s',x1)
#  logger.debug('x2:%s',x2)
#  logger.debug('pred:%s', pred)
#  logger.debug('y   :%s', y2)
#  logger.debug('统计%s,%s',np.min(x1, 0), np.max(x1, 0))
#  logger.debug('统计%s,%s',np.min(x2, 0), np.max(x2, 0))
  logger.info('holdout method准确率:%f', 1.0*sum(pred == y2)/len(y2))

#  pred2 = model.Predict(x1)
#  logger.debug(pred2)
#  logger.debug('holdout method准确率:%f', 1.0*sum(pred2 == y1)/len(y1))
#  logger.debug('tree:%s', model.root)
#
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

