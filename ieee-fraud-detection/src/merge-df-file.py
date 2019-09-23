# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import gc
import seaborn as sea
from sklearn import preprocessing
import xgboost as xgb
import sklearn

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

# Any results you write to the current directory are saved as output.


def mergeAllFile():
    file_trans_name = '../input/ieee-fraud-detection/train_transaction.csv'
    file_id_name = '../input/ieee-fraud-detection/train_identity.csv'
    file_output_name = 'train_all.csv'

    file_trans_name_test = '../input/ieee-fraud-detection/test_transaction.csv'
    file_id_name_test = '../input/ieee-fraud-detection/test_identity.csv'
    file_output_name_test = 'test_all.csv'
    
    mergefile(file_trans_name, file_id_name, file_output_name)
    mergefile(file_trans_name_test, file_id_name_test, file_output_name_test)

def mergefile(file_trans, file_id, file_output):
    trans = pd.read_csv(file_trans)
    id_df = pd.read_csv(file_id)
    trans.merge(id_df, on="TransactionID", how="left")
    trans.to_csv(file_output)

def encodeLabel(data, lbl):
    le = preprocessing.LabelEncoder()
    data[lbl] = le.fit_transform(data[lbl])
    return data

# mergeAllFile()
train = pd.read_csv("../input/train_all.csv")
test = pd.read_csv("../input/test_all.csv")
train_y = train['isFraud']
test_id = test['TransactionID']
attrs = ['TransactionDT','TransactionAmt','ProductCD',
    'card1','card2','card3','card4','card5','card6','addr1','addr2','dist1','dist2','P_emaildomain','R_emaildomain']

# Correlation ('V300','V309','V111','V124','V106','V125','V315','V134','V102','V123','V316','V113',
#              'V136','V305','V110','V299','V289','V286','V318','V304','V116','V284','V293',
#              'V137','V295','V301','V104','V311','V115','V109','V119','V321','V114','V133','V122','V319',
#              'V105','V112','V118','V117','V121','V108','V135','V320','V303','V297','V120',
#              'V1','V14','V41','V65','V88', 'V89', 'V107', 'V68', 'V28', 'V27', 'V29', 'V241','V269',
#              'V240', 'V325', 'V138', 'V154', 'V153', 'V330', 'V142', 'V195', 'V302', 'V328', 'V327', 
#              'V198', 'V196', 'V155')

train = train[attrs]
test = test[attrs]
bigx = train[attrs].append(test[attrs])
len_train = len(train)
#del train, test
#gc.collect()

for a in attrs:
    print(a, len(bigx[bigx[a].isnull()]))

#sea.countplot(train['ProductCD'])
bigx = encodeLabel(bigx, 'ProductCD')
#sea.countplot(bigx['ProductCD'])
bigx['hours'] = bigx['TransactionDT'] / 3600 % 24
bigx['weekday'] = bigx['TransactionDT'] / 3600 / 24 % 7

#print(bigx['hours'])
bigx = bigx.drop('TransactionDT', axis = 1)

#card1
c1mean = train['card1'].mean()
c1max = train['card1'].max()
c1min = train['card1'].min()
bigx['card1_mean'] = (bigx['card1'] - c1mean)/(c1max - c1min)
#sea.distplot(bigx['card1_mean'])
bigx = bigx.drop('card1', axis = 1)

#card2
c2max = train['card2'].max()
c2min = train['card2'].min()
bigx['card2_fill'] = bigx['card2'].map(lambda x : np.random.randint(c2min,c2max) if pd.isnull(x) else x)

sea.distplot(bigx['card2_fill'])
bigx = bigx.drop('card2', axis = 1)

#card3
sea.countplot(bigx['card3'])
train['card3'].value_counts()
bigx['card3_fill'] = bigx['card3'].map(lambda x : 150.0 if pd.isnull(x) else x)
bigx['card3_fill'].value_counts()
bigx = bigx.drop('card3', axis = 1)

#card4
train['card4'].value_counts()
bigx['card4_fill'] = bigx['card4'].map(lambda x : 'Unknown' if pd.isnull(x) else x)
bigx = encodeLabel(bigx, 'card4_fill')
bigx['card4_fill'].value_counts()
bigx = bigx.drop('card4', axis = 1)

#card5
sea.countplot(bigx['card5'])
train['card5'].value_counts()
bigx['card5_fill'] = bigx['card5'].map(lambda x : 226.0 if pd.isnull(x) else x)
bigx['card5_fill'].value_counts()
bigx = bigx.drop('card5', axis = 1)

#card6
train['card6'].value_counts()
bigx['card6_fill'] = bigx['card6'].map(lambda x : 'Unknown' if pd.isnull(x) else x)
bigx = encodeLabel(bigx, 'card6_fill')
bigx['card6_fill'].value_counts()
bigx = bigx.drop('card6', axis = 1)

#addr1
c2max = train['addr1'].max()
c2min = train['addr1'].min()
bigx['addr1_fill'] = bigx['addr1'].map(lambda x : np.random.randint(c2min,c2max) if pd.isnull(x) else x)

sea.distplot(bigx['addr1_fill'])
bigx = bigx.drop('addr1', axis = 1)

#addr2
sea.countplot(bigx['addr2'])
train['addr2'].value_counts()
bigx['addr2_fill'] = bigx['addr2'].map(lambda x : 87.0 if pd.isnull(x) else x)
bigx['addr2_fill'].value_counts()
bigx = bigx.drop('addr2', axis = 1)

#dist1
sea.countplot(bigx['dist1'])
train['dist1'].value_counts()
c2max = train['dist1'].max()
c2min = train['dist1'].min()
bigx['dist1_fill'] = bigx['dist1'].map(lambda x : np.random.randint(c2min,c2max) if pd.isnull(x) else x)
bigx = bigx.drop('dist1', axis = 1)

#dist2
sea.countplot(bigx['dist2'])
train['dist2'].value_counts()
c2max = train['dist2'].max()
c2min = train['dist2'].min()
bigx['dist2_fill'] = bigx['dist2'].map(lambda x : np.random.randint(c2min,c2max) if pd.isnull(x) else x)
bigx = bigx.drop('dist2', axis = 1)

#P_emaildomain
def binEmailDict(data, field_list, email_list_dict, field_NA_name):
    for f in field_list:
        for k in email_list_dict.keys():
            data[f + k] = 0
        data[f + field_NA_name] = 0
    for i, row in data.iterrows():
        for field in field_list:
            field_in_list = False
            if not isinstance(row[field],float) :
                if not pd.isnull(row[field]) :
                    for key,value in email_list_dict.items():
                        if row[field] in value:            
                            data.at[i, field + key] = 1
                            field_in_list = True
                            break
            if not field_in_list:
                data.at[i, field + field_NA_name] = 1
                
dict1 = {
    "hasYahoo" : ["yahoo.fr", "yahoo.de", "yahoo.es", "yahoo.co.uk", "yahoo.com", "yahoo.com.mx", "ymail.com", "rocketmail.com", "frontiernet.net"],
    "hasMSMail" : ["hotmail.com", "live.com.mx", "live.com", "msn.com", "hotmail.es", "outlook.es", "hotmail.fr", "hotmail.de", "hotmail.co.uk"],
    "hasMac" : ["icloud.com", "mac.com", "me.com"],
    "hasProdigy" : ["prodigy.net.mx", "att.net", "sbxglobal.net"],
    "hasCenturyLink" : ["centurylink.net", "embarqmail.com", "q.com"],
    "hasAIM" : ["aim.com", "aol.com"],
    "hasTWC" : ["twc.com", "charter.com"],
    "hasProton" : ["protonmail.com"],
    "hasComCast" : ["comcast.net"],
    "hasGmail" : ["gmail.com"],
    "hasAnonymous" : ["anonymous.com"],
}

#d = bigx[0:5].copy()
binEmailDict(bigx, ['P_emaildomain','R_emaildomain'], dict1, "hasNA")
bigx = bigx.drop("P_emaildomain", axis = 1)
bigx = bigx.drop("R_emaildomain", axis = 1)

#d = d.drop('P_emaildomain', axis = 1)
#d.columns.values.tolist()
print(bigx.columns.values.tolist())
# training
trainx = bigx[0:len_train]
testx = bigx[len_train::]


gbm = xgb.XGBClassifier(max_depth=5,learning_rate=0.01,n_estimators=300).fit(trainx, train_y )
predictions= gbm.predict(testx)

# Submit
submission = pd.DataFrame({ 'TransactionID': test_id,
                            'isFraud': predictions})
submission.to_csv("submission.csv", index=False)


#R_emaildomain