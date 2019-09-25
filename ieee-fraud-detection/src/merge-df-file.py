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

def encodeLabel(df, name):
    le = preprocessing.LabelEncoder()
    return pd.DataFrame({name : le.fit_transform(df)})
    

def encodeOneHot(df, lblprefix):
    return pd.get_dummies(df, prefix=lblprefix, dummy_na = True)
    
def saveColumn(df, nameprefix):
    df.to_csv("../input/" + "column_" + nameprefix + ".csv", index = None, header = True)    

def readColumn(nameprefix):
    return df.read_csv("../input/" + "column_" + nameprefix + ".csv")

def haveColumn(nameprefix):
    return os.path.exists("../input/" + "column_" + nameprefix + ".csv")

# mergeAllFile()
train = pd.read_csv("../input/train_all.csv")
test = pd.read_csv("../input/test_all.csv")
train_y = train['isFraud']
test_id = test['TransactionID']
attrs = ['TransactionDT','TransactionAmt','ProductCD',
    'card1','card2','card3','card4','card5','card6','addr1','addr2','dist1','dist2','P_emaildomain','R_emaildomain',
    'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14',
    'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15',
    'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9',
    'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 
    'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'V29',
    'V30', 'V31', 'V32', 'V33', 'V34', 'V35', 'V36', 'V37', 'V38', 'V39', 'V40', 'V41', 'V42', 'V43',
    'V44', 'V45', 'V46', 'V47', 'V48', 'V49', 'V50', 'V51', 'V52', 'V53', 'V54', 'V55', 'V56', 'V57',
    'V58', 'V59', 'V60', 'V61', 'V62', 'V63', 'V64', 'V65', 'V66', 'V67', 'V68', 'V69', 'V70', 'V71',
    'V72', 'V73', 'V74', 'V75', 'V76', 'V77', 'V78', 'V79', 'V80', 'V81', 'V82', 'V83', 'V84', 'V85',
    'V86', 'V87', 'V88', 'V89', 'V90', 'V91', 'V92', 'V93', 'V94', 'V95', 'V96', 'V97', 'V98', 'V99',
    'V100', 'V101', 'V102', 'V103', 'V104', 'V105', 'V106', 'V107', 'V108', 'V109', 'V110', 'V111', 'V112', 'V113', 
    'V114', 'V115', 'V116', 'V117', 'V118', 'V119', 'V120', 'V121', 'V122', 'V123', 'V124', 'V125', 'V126', 'V127', 
    'V128', 'V129', 'V130', 'V131', 'V132', 'V133', 'V134', 'V135', 'V136', 'V137', 'V138', 'V139', 'V140', 'V141', 
    'V142', 'V143', 'V144', 'V145', 'V146', 'V147', 'V148', 'V149', 'V150', 'V151', 'V152', 'V153', 'V154', 'V155', 
    'V156', 'V157', 'V158', 'V159', 'V160', 'V161', 'V162', 'V163', 'V164', 'V165', 'V166', 'V167', 'V168', 'V169',
    'V170', 'V171', 'V172', 'V173', 'V174', 'V175', 'V176', 'V177', 'V178', 'V179', 'V180', 'V181', 'V182', 'V183',
    'V184', 'V185', 'V186', 'V187', 'V188', 'V189', 'V190', 'V191', 'V192', 'V193', 'V194', 'V195', 'V196', 'V197',
    'V198', 'V199', 'V200', 'V201', 'V202', 'V203', 'V204', 'V205', 'V206', 'V207', 'V208', 'V209', 'V210', 'V211',
    'V212', 'V213', 'V214', 'V215', 'V216', 'V217', 'V218', 'V219', 'V220', 'V221', 'V222', 'V223', 'V224', 'V225',
    'V226', 'V227', 'V228', 'V229', 'V230', 'V231', 'V232', 'V233', 'V234', 'V235', 'V236', 'V237', 'V238', 'V239',
    'V240', 'V241', 'V242', 'V243', 'V244', 'V245', 'V246', 'V247', 'V248', 'V249', 'V250', 'V251', 'V252', 'V253',
    'V254', 'V255', 'V256', 'V257', 'V258', 'V259', 'V260', 'V261', 'V262', 'V263', 'V264', 'V265', 'V266', 'V267',
    'V268', 'V269', 'V270', 'V271', 'V272', 'V273', 'V274', 'V275', 'V276', 'V277', 'V278', 'V279', 'V280', 'V281',
    'V282', 'V283', 'V284', 'V285', 'V286', 'V287', 'V288', 'V289', 'V290', 'V291', 'V292', 'V293', 'V294', 'V295',
    'V296', 'V297', 'V298', 'V299', 'V300', 'V301', 'V302', 'V303', 'V304', 'V305', 'V306', 'V307', 'V308', 'V309',
    'V310', 'V311', 'V312', 'V313', 'V314', 'V315', 'V316', 'V317', 'V318', 'V319', 'V320', 'V321', 'V322', 'V323',
    'V324', 'V325', 'V326', 'V327', 'V328', 'V329', 'V330', 'V331', 'V332', 'V333', 'V334', 'V335', 'V336', 'V337',
    'V338', 'V339', 
    ]

loadattrs = []
for a in attrs:
    if not haveColumn(a):
        loadattrs.append(a)
# Correlation ('V300','V309','V111','V124','V106','V125','V315','V134','V102','V123','V316','V113',
#              'V136','V305','V110','V299','V289','V286','V318','V304','V116','V284','V293',
#              'V137','V295','V301','V104','V311','V115','V109','V119','V321','V114','V133','V122','V319',
#              'V105','V112','V118','V117','V121','V108','V135','V320','V303','V297','V120',
#              'V1','V14','V41','V65','V88', 'V89', 'V107', 'V68', 'V28', 'V27', 'V29', 'V241','V269',
#              'V240', 'V325', 'V138', 'V154', 'V153', 'V330', 'V142', 'V195', 'V302', 'V328', 'V327', 
#              'V198', 'V196', 'V155')
len_train = len(train)

train = train[loadattrs]
test = test[loadattrs]
bigx = train.append(test)

if 'card1' in loadattrs:
    c1mean = train['card1'].mean()
    c1max = train['card1'].max()
    c1min = train['card1'].min()
if 'card2' in loadattrs:
    c2max = train['card2'].max()
    c2min = train['card2'].min()
if 'addr1' in loadattrs:
    addr1max = train['addr1'].max()
    addr1min = train['addr1'].min()
if 'dist1' in loadattrs:
    dist1max = train['dist1'].max()
    dist1min = train['dist1'].min()
if 'dist2' in loadattrs:
    dist2max = train['dist2'].max()
    dist2min = train['dist2'].min()
del test
del train
gc.collect()

# for a in attrs:
#     print(a, len(bigx[bigx[a].isnull()]))

####sea.countplot(train['ProductCD'])
#bigx = encodeLabel(bigx, 'ProductCD')
if not haveColumn('ProductCD') :
    saveColumn( encodeOneHot(bigx['ProductCD'], 'ProductCD'), 'ProductCD' )
    bigx.drop('ProductCD', axis = 1, inplace=True)

####sea.countplot(bigx['ProductCD'])
if not haveColumn("TransactionDT"):
    df = pd.DataFrame()
    df['hours'] = bigx['TransactionDT'] / 3600 % 24
    df['weekday'] = bigx['TransactionDT'] / 3600 / 24 % 7
    saveColumn(df, 'TransactionDT')
    #print(bigx['hours'])
    bigx.drop('TransactionDT', axis = 1, inplace=True)
    print(bigx.columns.values.tolist())

#card1
if not haveColumn("card1"):
    saveColumn( (bigx['card1'] - c1mean)/(c1max - c1min), "card1")
    ###sea.distplot(bigx['card1_mean'])
    bigx.drop('card1', axis = 1, inplace = True)
    print(bigx.columns.values.tolist())

#card2
if not haveColumn("card2"):
    saveColumn( bigx['card2'].map(lambda x : np.random.randint(c2min,c2max) if pd.isnull(x) else x), 'card2')

    ###sea.distplot(bigx['card2_fill'])
    bigx.drop('card2', axis = 1, inplace = True)
    print(bigx.columns.values.tolist())

#card3
if not haveColumn("card3"):
    ##sea.countplot(bigx['card3'])
    #train['card3'].value_counts()
    saveColumn( bigx['card3'].map(lambda x : 150.0 if pd.isnull(x) else x), 'card3')
    #bigx['card3_fill'].value_counts()
    bigx.drop('card3', axis = 1, inplace = True)
    print(bigx.columns.values.tolist())

#card4
if not haveColumn("card4"):
    #train['card4'].value_counts()
    df = bigx['card4'].map(lambda x : 'Unknown' if pd.isnull(x) else x)
    saveColumn( encodeLabel(df, 'card4'), 'card4')
    #bigx['card4_fill'].value_counts()
    bigx.drop('card4', axis = 1, inplace = True)
    print(bigx.columns.values.tolist())

#card5
if not haveColumn("card5"):
    #seacountplot(bigx['card5'])
    #train['card5'].value_counts()
    saveColumn( bigx['card5'].map(lambda x : 226.0 if pd.isnull(x) else x), 'card5')
    #bigx['card5_fill'].value_counts()
    bigx.drop('card5', axis = 1, inplace = True)
    print(bigx.columns.values.tolist())

#card6
if not haveColumn("card6"):
    #train['card6'].value_counts()
    df = bigx['card6'].map(lambda x : 'Unknown' if pd.isnull(x) else x)
    saveColumn( encodeLabel(df, 'card6'), 'card6')
    #bigx['card6_fill'].value_counts()
    bigx.drop('card6', axis = 1, inplace = True)
    print(bigx.columns.values.tolist())

#addr1
if not haveColumn("addr1"):
    saveColumn( bigx['addr1'].map(lambda x : np.random.randint(addr1min,addr1max) if pd.isnull(x) else x) , 'addr1')

    ##seadistplot(bigx['addr1_fill'])
    bigx.drop('addr1', axis = 1, inplace = True)
    print(bigx.columns.values.tolist())

#addr2
if not haveColumn("addr2"):
    #seacountplot(bigx['addr2'])
    #train['addr2'].value_counts()
    saveColumn( bigx['addr2'].map(lambda x : 87.0 if pd.isnull(x) else x), 'addr2')
    #bigx['addr2_fill'].value_counts()
    bigx.drop('addr2', axis = 1, inplace = True)
    print(bigx.columns.values.tolist())

#dist1
if not haveColumn("dist1"):
    #seacountplot(bigx['dist1'])
    #train['dist1'].value_counts()

    saveColumn( bigx['dist1'].map(lambda x : np.random.randint(dist1min,dist1max) if pd.isnull(x) else x), 'dist1')
    bigx.drop('dist1', axis = 1, inplace = True)
    print(bigx.columns.values.tolist())

#dist2
if not haveColumn("dist2"):
    ###sea.countplot(bigx['dist2'])
    #train['dist2'].value_counts()

    saveColumn( bigx['dist2'].map(lambda x : np.random.randint(dist2min,dist2max) if pd.isnull(x) else x), 'dist2')
    bigx.drop('dist2', axis = 1, inplace = True)
    print(bigx.columns.values.tolist())

    bigx.to_csv("../input/dist2.csv", index = None, header = True)

#P_emaildomain
def binEmailDictIter(data, field_list, email_list_dict, field_NA_name):
    dict_map = dict()
    for f in field_list:
        for k in email_list_dict.keys():
            data[f + k] = 0
        data[f + field_NA_name] = 0    
    print(data.columns.values.tolist())
    for key,value in email_list_dict.items():
        for v in value:
            dict_map[v] = key
    rowcount = 0
    for i, row in data.iterrows():
        rowcount += 1
        if rowcount % 10000 == 0 :
            print(rowcount)
        for field in field_list:
            field_in_list = False
            if not isinstance(row[field],float) :
                if not pd.isnull(row[field]) :
                    if row[field] in dict_map:
                        data.at[i, field + dict_map[row[field]]] = 1
                        field_in_list = True
                        break
            if not field_in_list:
                data.at[i, field + field_NA_name] = 1

def binEmailDict(data, field_list, email_list_dict, field_NA_name):
    dict_map = dict()
    for key,value in email_list_dict.items():
        for v in value:
            dict_map[v] = key
    for field in field_list:
        data[field]=data[field].map(dict_map)
        data = encodeOneHot(data, field, field)
    return data
if not haveColumn("P_emaildomain") or not haveColumn("R_emaildomain"):
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
    dict_map = dict()
    for key,value in dict1.items():
        for v in value:
            dict_map[v] = key
    for field in ['P_emaildomain','R_emaildomain']:
        df = bigx[field].map(dict_map)
        saveColumn( encodeOneHot(df, field), field)
    #d = bigx[0:5].copy()
    #bigx = binEmailDict(bigx, ['P_emaildomain','R_emaildomain'], dict1, "hasNA")

    # title_list = bigx.columns.values.tolist()
    # for x in title_list:
    #     if x.startswith("P_emaildomain_has") or x.startswith("R_emaildomain_has"):
    #         print(x, bigx[x].sum())

    bigx.drop("P_emaildomain", axis = 1, inplace = True)
    bigx.drop("R_emaildomain", axis = 1, inplace = True)

#C1,C2,C3
C_list = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14']
for c_field in C_list:
    if not haveColumn(c_field):
        saveColumn(bigx[c_field].fillna(0), c_field)
        bigx.drop(c_field, axis = 1, inplace = True)
#bigx['C2'].fillna(1.0,inplace=True)
#bigx['C3'].fillna(0.0,inplace=True)
#D1,D2,D3
D_list = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15']
for D_field in D_list:
    if not haveColumn(D_field):
        saveColumn(bigx[D_field].fillna(-999), D_field)
        bigx.drop(D_field, axis = 1, inplace = True)

M_list = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9']
for M_field in M_list:
    if not haveColumn(M_field):
        saveColumn(encodeOneHot(bigx[M_field], M_field), M_field)
        bigx.drop(M_field, axis = 1, inplace = True)


for i in range(1,340):
    V_field = 'V' + str(i)
    if not haveColumn(V_field):
        saveColumn(bigx[V_field].fillna(-999), V_field)
        bigx.drop(V_field, axis = 1, inplace = True)

del bigx
gc.collect()

bigx = pd.DataFrame()

for att in attrs:
    bigx = pd.concat([bigx, readColumn(att)], axis = 1)

bigx.to_csv("../input/train_final.csv", index = None, header = True)

#d = d.drop('P_emaildomain', axis = 1, inplace = True)
#d.columns.values.tolist()
print(bigx.columns.values.tolist())
# training
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


#R_emaildomain