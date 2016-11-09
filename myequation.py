#!/usr/bin/python
#coding:utf-8

import numpy as np
np.seterr(divide ='ignore')
import math
def Entropy(vector_class):
  m = np.sum(vector_class)
  if m <= 0.1:return 0.0

  vec = 1.0/m*np.array(vector_class)
#  print vec
  try:
    e = -1.0 * np.dot(vec, np.nan_to_num(np.log2(vec)))
  except RuntimeWarning:
    pass
  return e

#2分
def InformationGain(ig_parent, vector_split1_class, vector_split2_class):
  e_split1 = Entropy(vector_split1_class)
  e_split2 = Entropy(vector_split2_class)
#  print 'e1,e2',e_split1,e_split2

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
  tmp = 1.0 - np.dot(np.array(vector_class) / total, np.array(vector_class) / total)
  return tmp

def ClassError(vector_class):
  total = float(sum(vector_class))
  return (float(total) - max(vector_class))/total

if __name__ == '__main__':
  print '%.3f'%(InformationGain(0.991, [0,6,8], [0,6,8]))
  print '%.3f'%(Gini([0,6,8]))
  print '%.3f'%(ClassError([0,6,8]))
  print '%f'%(Entropy([0,0,4]))

