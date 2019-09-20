import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import sklearn

#import strings
def substrings_in_string(big_string, substrings):
    for substring in substrings:
        if str.find(big_string, substring) != -1:
            return substring
    print(big_string)
    return np.nan

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

attrs = ['Pclass', 'Sex', 'Age', 'Fare', "Parch", 'SibSp', 'Embarked', 'Name', 'Cabin']
bigx = train[attrs].append(test[attrs])

for att in attrs:
    print(att,bigx[att].isnull().sum())
# Name column
title_list=['Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev',
                    'Dr', 'Ms', 'Mlle','Col', 'Capt', 'Mme', 'Countess',
                    'Don', 'Jonkheer']
bigx['Title'] = bigx['Name'].map(lambda x: substrings_in_string(x, title_list))


# replacing all titles with mr, mrs, miss, master
def replace_titles(x):
    title = x['Title']
    if title in ['Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col']:
        return 'Mr'
    elif title in ['Countess', 'Mme']:
        return 'Mrs'
    elif title in ['Mlle', 'Ms']:
        return 'Miss'
    elif title == 'Dr':
        if x['Sex'] == 'Male':
            return 'Mr'
        else:
            return 'Mrs'
    else:
        return title


bigx['Title'] = bigx.apply(replace_titles, axis=1)
le = LabelEncoder()
bigx['Title'] = le.fit_transform(bigx['Title'])
bigx = bigx.drop('Name', axis=1)

# Cabin
# Turning cabin number into Deck

#bigx = train[attrs].append(test[attrs])
bigx['Cabin']=bigx['Cabin'].map(lambda x: 'Unknown' if pd.isnull(x) else x)
# bigx['2']=bigx['Cabin']
# nullcabin = bigx[bigx['Cabin'].isnull()]
# for index, row in nullcabin.iterrows():
#     #if(index > 10):
#     #    break
#     print(index)
#     bigx.at[index, '2'] = 'Unknown'

cabin_list = ['A', 'B', 'C', 'D', 'E', 'F', 'T', 'G', 'Unknown']
bigx['Deck'] = bigx['Cabin'].map(lambda x: substrings_in_string(x, cabin_list))
bigx['Deck'] = le.fit_transform(bigx['Deck'])
bigx = bigx.drop('Cabin', axis = 1)

# Fare column
nullfare = bigx[bigx['Fare'].isnull()]
bigx.at[nullfare.index,'Fare']= bigx['Fare'].median()

# Sex column
#le = LabelEncoder()
bigx['Sex'] = le.fit_transform(bigx['Sex'])

# Embarked column
embcount = bigx['Embarked'].value_counts()
embnan = bigx[bigx['Embarked'].isnull()]
newembark = np.random.choice(['S','C','Q'],len(embnan))
idx = 0
for index, row in embnan.iterrows():
    # print(index, row)
    bigx.at[index,'Embarked'] = newembark[idx]
    idx += 1
bigx['Embarked'] = le.fit_transform(bigx['Embarked'])

# Age column
agecount = bigx['Age'].value_counts()
dist = agecount / agecount.sum()
newage = np.random.choice(agecount.index.tolist(), len(bigx[bigx['Age'].isnull()]), dist.values.tolist())

i = 0
for index,row in bigx[bigx['Age'].isnull()].iterrows() :
    bigx.at[index, 'Age'] = newage[i]
    i+=1

# Training

trainx = bigx[0:len(train)]
testx = bigx[len(train)::]
train_y = train['Survived']

gbm = xgb.XGBClassifier(max_depth=3,learning_rate=0.01,n_estimators=300).fit(trainx, train_y )
predictions= gbm.predict(testx)

# Submit
submission = pd.DataFrame({ 'PassengerId': test['PassengerId'],
                            'Survived': predictions})
submission.to_csv("submission.csv", index=False)

#truth = pd.read_csv('../input/gender_submission.csv')
#truthcol = truth['Survived']

#sklearn.metrics.accuracy_score(truthcol, predictions)