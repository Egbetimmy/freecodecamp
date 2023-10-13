import numpy as np

def calculate(list):
  """return mean, variance, standard deviation, max, min, and sum of the rows, columns, and elements in a 3 x 3 matrix.
  list: containing 9 digits"""
  if len(list)!=9:
    raise ValueError("List must contain nine numbers.")
  ar = np.array(list).reshape((3,3))
  
  # empty dict. add entries with update()
  calculations = {}

  # tolist()
  # ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
  # https://sopython.com/canon/119/the-truth-value-of-an-array-with-more-than-one-element-is-ambiguous/
  # print( type(ar.mean(0)) ) # <class 'numpy.ndarray'>
  # print( type(ar.mean(0).tolist()) ) # <class 'list'>

  # mean or average: sum of values / number of values
  # median: middle value when a data set is ordered from least to greatest
  # mode: number that occurs most often
  calculations.update(mean=[ar.mean(0).tolist(), ar.mean(1).tolist(), ar.mean(None).tolist()])

  # variance (wikipedia):
  # variance is the expectation of the squared deviation of a random variable from its mean
  # Informally, it measures how far a set of numbers are spread out from their average
  # The variance is the square of the standard deviation
  calculations.update(variance=[ar.var(0).tolist(), ar.var(1).tolist(), ar.var(None).tolist()])

  # standard deviation (wikipedia)
  # standard deviation is a measure of the amount of variation or dispersion of a set of values.
  # A low standard deviation indicates that the values tend to be close to the mean (also called the expected value) of the set,
  # while a high standard deviation indicates that the values are spread out over a wider range
  calculations.update({'standard deviation':[ar.std(0).tolist(), ar.std(1).tolist(), ar.std(None).tolist()]})

  # max
  # np.amax(ar, 0|1|None) = ar.max(0|1|None)
  # 0: max in column 
  # 1: max in row 
  # None: max of all ('flattened')
  #calculations.update(max=[np.amax(ar, 0), np.amax(ar, 1), np.amax(ar, None)])
  calculations.update(max=[ar.max(0).tolist(), ar.max(1).tolist(), ar.max(None).tolist()])

  # min
  calculations.update(min=[ar.min(0).tolist(), ar.min(1).tolist(), ar.min(None).tolist()])

  # sum
  calculations.update(sum=[ar.sum(0).tolist(), ar.sum(1).tolist(), ar.sum(None).tolist()])
  
  #print(calculations)
  return calculations