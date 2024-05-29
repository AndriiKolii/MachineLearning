import pandas as pd
from sklearn import *
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel

days2 = pd.read_csv('DaysInHospital_Y2.csv', index_col='MemberID')
members = pd.read_csv('Members.csv', index_col='MemberID')
claims = pd.read_csv('Claims_Y1.csv', index_col='MemberID')

i = pd.notnull(members.AgeAtFirstClaim)
members.loc[i, 'AgeAtFirstClaim'] = members.loc[i, 'AgeAtFirstClaim'].apply(lambda s: s.split('-')[0] if s != '80+' else '80')
members.loc[i, 'AgeAtFirstClaim'] = members.loc[i, 'AgeAtFirstClaim'].apply(lambda s: int(s))

members.AgeAtFirstClaim = members.AgeAtFirstClaim.fillna(value=-1)

members.Sex = members.Sex.fillna(value='N')

claims.LengthOfStay = claims.LengthOfStay.fillna(value=0)

claims.CharlsonIndex = claims.CharlsonIndex.map({'0':0, '1-2':1, '3-4':3, '5+':5})

claims.LengthOfStay = claims.LengthOfStay.map({'0':0, '1 day':1, '2 days':2, '3 days':3, '4 days':4, '5 days':5,
                                               '6 days':6, '1- 2 weeks':10, '2- 4 weeks':21, '4- 8 weeks':42, '26+ weeks':182})

f_Charlson = claims.groupby(['MemberID']).CharlsonIndex.max()
f_LengthOfStay = claims.groupby(['MemberID']).LengthOfStay.sum()

data = days2.copy()
data = data.join(f_Charlson)
data = data.join(f_LengthOfStay)
data = data.join(members['AgeAtFirstClaim'])

data = pd.get_dummies(data, columns=['ClaimsTruncated'], prefix='ClaimTruncated')
data = pd.get_dummies(data, columns=['CharlsonIndex'], prefix='f_Charlson')
data = pd.get_dummies(data, columns=['LengthOfStay'], prefix='LengthOfStay')

data = data.join(pd.get_dummies(members['Sex'], prefix='pol'))

x = data.drop('DaysInHospital', axis=1)
y = data['DaysInHospital']

selector = SelectFromModel(estimator=linear_model.LogisticRegression(penalty='l1', solver='liblinear', random_state=42))
x_new = selector.fit_transform(x, y)
new_feature = x.columns[selector.get_support()]


def calcAUC(data):
    dataTrain = data.sample(frac=0.8, random_state=42)
    dataTest = data.drop(dataTrain.index)
    model = linear_model.LogisticRegression()

    # model.fit( dataTrain.loc[:, dataTrain.columns != 'DaysInHospital'], dataTrain.DaysInHospital )
    # predictionProb = model.predict_proba( dataTest.loc[:, dataTest.columns != 'DaysInHospital'] )

    model.fit(dataTrain[new_feature], dataTrain.DaysInHospital)
    predictionProb = model.predict_proba(dataTest[new_feature])

    fpr, tpr, _ = metrics.roc_curve(dataTest['DaysInHospital'], predictionProb[:,1])
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.show()
    print(metrics.roc_auc_score(dataTest['DaysInHospital'], predictionProb[:,1]) )

calcAUC(data)

print(new_feature)

# print(data.head(30))
