import multiprocessing
import pandas as pd

files=['../input/sample_submission.csv',
    '../input/test_identity.csv',
    '../input/test_transaction.csv',
    '../input/train_identity.csv',
    '../input/train_transaction.csv'
    ]

#%%time

def load_data(file):
    return pd.read_csv(file)

with multiprocessing.Pool() as pool:
    sub, test_id, test_tr, train_id, train_tr = pool.map(load_data,files)

