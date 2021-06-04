'''Data loader for UCI letter, spam and MNIST datasets.
'''

# Necessary packages
import numpy as np
import pandas as pd
from utils import binary_sampler
from scipy.io import arff
from sklearn.preprocessing import LabelEncoder
# from keras.datasets import mnist


# def data_loader (data_name, miss_rate):
#   '''Loads datasets and introduce missingness.
#
#   Args:
#     - data_name: letter, spam, or mnist
#     - miss_rate: the probability of missing components
#
#   Returns:
#     data_x: original data
#     miss_data_x: data with missing values
#     data_m: indicator matrix for missing components
#   '''
#
#   # Load data
#   if data_name in ['letter', 'spam']:
#     file_name = 'data/'+data_name+'.csv'
#     data_x = np.loadtxt(file_name, delimiter=",", skiprows=1)
#   elif data_name == 'mnist':
#     (data_x, _), _ = mnist.load_data()
#     data_x = np.reshape(np.asarray(data_x), [60000, 28*28]).astype(float)
#
#   # Parameters
#   no, dim = data_x.shape
#
#   # Introduce missing data
#   data_m = binary_sampler(1-miss_rate, no, dim)
#   miss_data_x = data_x.copy()
#   miss_data_x[data_m == 0] = np.nan
#
#   return data_x, miss_data_x, data_m

def data_loader(data_name):
  if data_name in ['spam', 'letter','breast','news','credit']:
    file_name = 'data/' + data_name + '_full.csv'
    df = pd.read_csv(file_name)
    x = df.drop(['target'],axis = 1)
    x = x.values
    y = df['target'].values
    if data_name == 'letter':
      le = LabelEncoder()
      y = le.fit_transform(y)

  else:
    file_name = 'data/' + data_name + '.arff'
    data, _ = arff.loadarff(file_name)
    data = data.tolist()
    x = np.array([item[:-1] for item in data])
    y = np.array([item[-1] for item in data])
    le = LabelEncoder()
    y = le.fit_transform(y)
  # Parameters
  no, dim = x.shape
  print("Num of samples:", no)
  print("Num of feature:", dim)
  print("Num of classes:", len(set(y)))
  x = x.astype(np.float32)
  return x, y

def make_missing_data(x, miss_rate, seed = 42):
  no, dim = x.shape
  m = binary_sampler(1 - miss_rate, no, dim, seed=seed)
  miss_x = x.copy()
  miss_x[m == 0] = np.nan
  return miss_x, m