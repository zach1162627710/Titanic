import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cross_validation import train_test_split


train = pd.read_csv('D:/Kaggle/titanic/train.csv')
# print(train.head(5))
test = pd.read_csv('D:/Kaggle/titanic/test.csv')
# print('train_data:',train.shape,'test_data:',test.shape)
rownum_train = train.shape[0]
rownum_test = test.shape[0]
print("训练集行数:",rownum_train,"测试集行数",rownum_test)

full = train.append( test , ignore_index = True )
# print("总行数：",full.shape[0])
# print(full.describe())
# print(full.info())

#数据清洗，
    # #数值型，mean取代
full['Age']=full['Age'].fillna(full['Age'].mean())
full['Fare']=full['Fare'].fillna(full['Fare'].mean())
# print(full.info())
    #直接分类型，缺失较少的用最常见类别取代
# print(full['Embarked'].head(10))
print(full["Embarked"].value_counts())
full['Embarked'] = full["Embarked"].fillna('S')
full["Cabin"] = full['Cabin'].fillna('UN')
# print(full.info())

#特征提取
    #one-hot编码，整合
EmbarkedDF = pd.DataFrame()
EmbarkedDF = pd.get_dummies(full['Embarked'],prefix='Embarked')   #one-hot编码，dummy variables
full = pd.concat([full, EmbarkedDF],axis=1)
full.drop('Embarked',axis = 1,inplace = True)

pclassDf = pd.DataFrame()
pclassDf = pd.get_dummies(full['Pclass'] , prefix='Pclass')
full = pd.concat([full,pclassDf],axis=1)
full.drop('Pclass',axis=1,inplace=True)


#map映射
sex_mapDcit= {'male':1,'female':0}
full['Sex']=full['Sex'].map(sex_mapDcit)
# print(full["Sex"].head(5))

#根据名称提取特征
# print(full['Name'].head(5))
def getTitle(name):
    str1 = name.split(',')[1]
    str2 = str1.split('.')[0]
    str3 = str2.strip()
    return str3
titleDF = pd.DataFrame()
titleDF['Title'] = full['Name'].map(getTitle)
print(titleDF.head(5))
title_mapDict ={
                    "Capt":       "Officer",
                    "Col":        "Officer",
                    "Major":      "Officer",
                    "Jonkheer":   "Royalty",
                    "Don":        "Royalty",
                    "Sir" :       "Royalty",
                    "Dr":         "Officer",
                    "Rev":        "Officer",
                    "the Countess":"Royalty",
                    "Dona":       "Royalty",
                    "Mme":        "Mrs",
                    "Mlle":       "Miss",
                    "Ms":         "Mrs",
                    "Mr" :        "Mr",
                    "Mrs" :       "Mrs",
                    "Miss" :      "Miss",
                    "Master" :    "Master",
                    "Lady" :      "Royalty"
}
titleDF['Title'] = titleDF['Title'].map(title_mapDict)
    #名称Title编码
titleDF = pd.get_dummies(titleDF['Title'])
full = pd.concat([full,titleDF],axis = 1)
full.drop('Name',axis = 1,inplace = True)
# print(full.head(5))
    #船厂编码
cabinDF = pd.DataFrame()
full['Cabin'] = full["Cabin"].map(lambda c:c[0])
cabinDF = pd.get_dummies(full['Cabin'],prefix='Cabin')
# print(cabinDF.head())
full = pd.concat([full,cabinDF],axis = 1)
full.drop('Cabin',axis= 1,inplace=True)
# print(full.columns)
    #family lambda表达式编码
#family_num
familyDF = pd.DataFrame()
familyDF['FamilySize']=full['Parch']+full['SibSp']+1
# print(familyDF['FamilySize'].head())
familyDF['familysmall'] = familyDF['FamilySize'].map(lambda s:1 if s<=1 else 0)
familyDF['familymiddle'] = familyDF['FamilySize'].map(lambda s:1 if (s>=2 & s<=4) else 0)
familyDF['familylarge'] = familyDF['FamilySize'].map(lambda s:1 if s>=5 else 0)

full = pd.concat([full,familyDF],axis = 1)
# print(full.head())
# print(full.shape)

corrmat = full.corr()
f, ax = plt.subplots(figsize=(9, 9))
sns.heatmap(corrmat, vmax=1, square=True)
plt.show()

k = 20
df = corrmat.nlargest(k, 'Survived')
cols = df['Survived'].index

cm = np.corrcoef(full[cols].values.T)
# print(cm)
sns.set(font_scale=1)
hm = sns.heatmap(cm, cbar=True, annot=True, \
                 square=True, fmt='.2f', annot_kws={'size': 5}, yticklabels=cols.values, xticklabels=cols.values)
# plt.show()

full_X = pd.concat( [titleDF,       #头衔
                     pclassDf,      #客舱等级
                     familyDF,      #家庭大小
                     full['Fare'],  #船票价格
                         cabinDF,   #船舱号
                     EmbarkedDF,    #登船港口
                     full['Sex']    #性别
                    ] , axis=1 )

source_X = full_X.loc[0:rownum_train-1,:]

source_y = full.loc[0:rownum_train-1,'Survived']

pred_X = full_X.loc[rownum_train:,:]
'''
确保这里原始数据集取的是前891行的数据，不然后面模型会有错误
'''
print('原始数据集有多少行:',source_X.shape[0])
print('测试数据集有多少行:',pred_X.shape[0])


#建立模型用的训练数据集和测试数据集
train_X, test_X, train_y, test_y = train_test_split(source_X ,
                                                    source_y,
                                                    train_size=.8)
print ('原始数据集特征：',source_X.shape,
       '训练数据集特征：',train_X.shape ,
      '测试数据集特征：',test_X.shape)

print ('原始数据集标签：',source_y.shape,
       '训练数据集标签：',train_y.shape ,
      '测试数据集标签：',test_y.shape)

#创建模型
# from sklearn.linear_model import LogisticRegression
# model = LogisticRegression()

# from sklearn.ensemble import RandomForestClassifier
# model = RandomForestClassifier(n_estimators=100)

# from sklearn.svm import SVC, LinearSVC
# model = SVC()

from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors = 3)
model.fit( train_X , train_y )
score = model.score(test_X , test_y )
print(score)


