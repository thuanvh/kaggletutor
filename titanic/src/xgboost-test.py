import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import sklearn

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

attrs = ['Pclass', 'Sex', 'Age', 'Fare', "Parch", 'SibSp', 'Embarked']
bigx = train[attrs].append(test[attrs])

for att in attrs:
    print(att,bigx[att].isnull().sum())

nullfare = bigx[bigx['Fare'].isnull()]
bigx.at[nullfare.index,'Fare']= bigx['Fare'].median()

le = LabelEncoder()
bigx['Sex'] = le.fit_transform(bigx['Sex'])

embcount = bigx['Embarked'].value_counts()
embnan = bigx[bigx['Embarked'].isnull()]
newembark = np.random.choice(['S','C','Q'],len(embnan))
idx = 0
for index, row in embnan.iterrows():
    # print(index, row)
    bigx.at[index,'Embarked'] = newembark[idx]
    idx += 1
bigx['Embarked'] = le.fit_transform(bigx['Embarked'])

agecount = bigx['Age'].value_counts()
dist = agecount / agecount.sum()
newage = np.random.choice(agecount.index.tolist(), len(bigx[bigx['Age'].isnull()]), dist.values.tolist())

i = 0
for index,row in bigx[bigx['Age'].isnull()].iterrows() :
    bigx.at[index, 'Age'] = newage[i]
    i+=1

trainx = bigx[0:len(train)]
testx = bigx[len(train)::]
train_y = train['Survived']

gbm = xgb.XGBClassifier(max_depth=3,learning_rate=0.05,n_estimators=300).fit(trainx, train_y )
predictions= gbm.predict(testx)

submission = pd.DataFrame({ 'PassengerId': test['PassengerId'],
                            'Survived': predictions})
submission.to_csv("submission.csv", index=False)

#truth = pd.read_csv('../input/gender_submission.csv')
#truthcol = truth['Survived']

#sklearn.metrics.accuracy_score(truthcol, predictions)