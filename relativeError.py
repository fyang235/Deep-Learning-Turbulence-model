import numpy as np

def rel_maxError(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))
  
def rel_meanError(x, y):
  """ returns relative error """
  return np.mean(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))