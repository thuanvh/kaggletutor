import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import gc
import seaborn as sea
from sklearn import preprocessing
import xgboost as xgb
import sklearn

bigx = pd.read_csv("../input/train_final.csv")
train_y = pd.read_csv("../input/isFraud.csv")
test_id = pd.read_csv("../input/testId.csv")

len_train = len(train_y)
train_y = train_y['isFraud']
test_id = test_id['TransactionID']

trainx = bigx[0:len_train]
testx = bigx[len_train::]

#train_y = pd.read_csv("../input/train_all.csv", names=["isFraud"])

print('Training on', len(trainx), 'samples', 'Number of isFraud:', train_y.sum())
gbm = xgb.XGBClassifier(max_depth=5,learning_rate=0.05,n_estimators=500).fit(trainx, train_y )
predictions= gbm.predict(testx)

# Submit
submission = pd.DataFrame({ 'TransactionID': test_id,
                            'isFraud': predictions})
submission.to_csv("submission.csv", index=False)
