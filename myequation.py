#!/usr/bin/python
#coding:utf-8

import numpy as np
import math
def Entropy(vector_class):
  num_class = len(vector_class)

  if np.count_nonzero(vector_class) < num_class:
    return 0.0

  m = np.sum(vector_class)
  if m == 0.0:return 0.0

  e = -np.dot(1.0 / m * np.array(vector_class), 
      np.log2(1.0 / m * np.array(vector_class)))
  return e

#2分
def InformationGain(ig_parent, vector_split1_class, vector_split2_class):
  e_split1 = Entropy(vector_split1_class)
  e_split2 = Entropy(vector_split2_class)
  print 'e1,e2',e_split1,e_split2

  num_split1 = np.sum(vector_split1_class)
  num_split2 = np.sum(vector_split2_class)
  N = num_split1 + num_split2
#  print N

  v1 = np.array([num_split1, num_split2])
  v2 = np.array([e_split1, e_split2])

  ig = ig_parent - 1.0 / N * np.dot(v1, v2)

  return ig

def Gini(vector_class):
  total = float(sum(vector_class))
  tmp = pow(total, 2) - pow(float(vector_class[0]), 2) - pow(float(vector_class[1]), 2)
  return tmp / pow(total, 2)

def ClassError(vector_class):
  total = float(sum(vector_class))
  return (float(total) - max(vector_class))/total

if __name__ == '__main__':
  print '%.3f'%(InformationGain(0.991, [4,4, 4], [0,1, 3]))
  print '%.3f'%(Gini([0,6]))
  print '%.3f'%(ClassError([0,6]))
  param = {}
  param['gini'] = Gini
  param['gini']([0,6])
