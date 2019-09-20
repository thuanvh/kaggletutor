import pandas as pd
import numpy as np
import multiprocessing
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import gc
from time import time
import datetime
from tqdm import tqdm_notebook
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, KFold, TimeSeriesSplit, train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import graphviz
warnings.simplefilter('ignore')
sns.set()
#%matplotlib inline

files = ['../input/test_identity.csv', 
         '../input/test_transaction.csv',
         '../input/train_identity.csv',
         '../input/train_transaction.csv',
         '../input/sample_submission.csv']

#%%time
def load_data(file):
    return pd.read_csv(file)

#with multiprocessing.Pool() as pool:
#    test_id, test_tr, train_id, train_tr, sub = pool.map(load_data, files)

#with multiprocessing.Pool() as pool:
print(files[0])
test_id = load_data(files[0])
print(files[1])
test_tr = load_data(files[1])
print(files[2])
train_id = load_data(files[2])
print(files[3])
train_tr = load_data(files[3])
print(files[4])
sub = load_data(files[4])