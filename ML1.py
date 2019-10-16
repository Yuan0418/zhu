#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 16:34:36 2019
@author: yuanzhu
"""
import pandas as pd
from sklearn import preprocessing

train_set1 = pd.read_csv('/Users/yuanzhu/Downloads/tcdml1920-income-ind/training.csv')
test_set1 = pd.read_csv('/Users/yuanzhu/Downloads/tcdml1920-income-ind/prediction.csv')
#age height in a range replace by mean
train_set1.drop_duplicates()#drop same rows

#print(train_set1.info())
yy = train_set1["Income"]
train_set1["Income1"]=yy
train_set1["Income"] = preprocessing.scale(train_set1["Income"])
train_set1 = train_set1.drop(train_set1[(train_set1.Income > 3)].index)
print(train_set1.head())
y = train_set1["Income1"]
train_set1=train_set1.drop('Income1', axis=1)
len=train_set1.shape[0]
print(train_set1.head())

merge_data=pd.concat([train_set1,test_set1])

merge_data=merge_data.drop('Income', axis=1)
train_set=merge_data.drop('Instance', axis=1)
train_set=train_set.drop('Glasses', axis=1)
train_set=train_set.drop('HairColor', axis=1)

print(train_set.info())#185046 185212
print(train_set.head())
#Analyze Data
####one hot####
#数字类型选择出来，并将其中有缺失值的部分进行填充
train_set.Age.fillna(train_set.Age.median(),inplace=True)
train_set.Year.fillna(train_set.Year.median(),inplace=True)
#delete unuseful cols hiar height


#train_set["Age"] = preprocessing.scale(train_set["Age"])
#train_set["Year"] = preprocessing.scale(train_set["Year"])
#train_set["BodyHeight"] = preprocessing.scale(train_set["BodyHeight"])
"""
num_cols=train_set[['Year','Age','Size','BodyHeight']]
train_set=train_set.drop('Year', axis=1)
train_set=train_set.drop('Age', axis=1)
train_set=train_set.drop('Size', axis=1)
train_set=train_set.drop('BodyHeight', axis=1)
min_max_scaler = preprocessing.MinMaxScaler()
std_x = preprocessing.StandardScaler().fit_transform(num_cols)
std_xx= pd.DataFrame(std_x,columns=list('ABCD'))
print(std_xx.head())
train_set=pd.concat([train_set, std_xx], axis=1, join_axes=[train_set.index])
print(train_set.head())
"""

train_set.Gender.fillna(method='pad') ## 用前一个数据代替NaN：method='pad'
train_set = pd.get_dummies(train_set, columns=['Gender'],prefix='Gender') #one hot encoding
train_set.Degree.fillna(method='pad')
train_set = pd.get_dummies(train_set, columns=['Degree'],prefix='Degree')
train_set = pd.get_dummies(train_set, columns=['Country'],prefix='Country')
train_set.Profession.fillna(method='pad')
train_set = pd.get_dummies(train_set, columns=['Profession'],prefix='Profession')

X_train=train_set.head(len)
X_test=train_set.tail(73230)

print(X_train.info())
print(X_test.info())

print("######train set cleaning ok#######")


print("######moedel#######")

#from sklearn.model_selection import train_test_split
# 避免过拟合，采用交叉验证，验证集占训练集20%，固定随机种子（random_state)
#train_X,test_X, train_y, test_y = train_test_split(X_train, y, test_size = 0.2,random_state = 0)

from sklearn import linear_model
linear = linear_model.LinearRegression()
linear.fit(X_train,y)
pred=linear.predict(X_train)
print("score: ",linear.score(X_train,y))
from sklearn.metrics import mean_squared_error
mse=mean_squared_error(pred,y)   #使用均方误差来评价模型好坏，可以输出mse进行查看评价值
print("mse: ",mse)

"""
print("######test model#######")
pred1=linear.predict(test_X)
print("score: ",linear.score(test_X,test_y))
mse=mean_squared_error(pred1,test_y)   #使用均方误差来评价模型好坏，可以输出mse进行查看评价值
print("mse: ",mse)
"""

#print("######test pred#######")
#pred2=linear.predict(X_test)
#dataframe = pd.DataFrame({'income_pred':pred2})
#dataframe.to_csv(r"2256.csv",sep=',')
