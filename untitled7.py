# -*- coding: utf-8 -*-
"""
Created on Wed Dec  17 15:44:11 2020

@author: goodm
"""

import pandas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, LinearRegression
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier


df = pandas.read_excel('aaa/Terminated_Contracts_Surgut2020.xlsx')
df1 = pandas.read_excel('aaa/Copy_Accepted_Values_Rostelecom.xlsx')

#показывает на 2 меньше индексы из-за пайтона 

max1 = max(df['Дата снятия'] - df['Дата договора'])
min0 = min(df['Дата снятия'] - df['Дата договора'])

#print(max1)
#print(min0)

#df = df.assign(q=pd.Series(np.random.randn(229)).values)
#df = df.assign(r=pd.Series(np.random.randn(229)).values)
df['Количество дней договора'] = pd.Series(np.random.randn(11718), index=df.index)



i=0
for i in range(11718):
    df['Количество дней договора'][i] = (df['Дата снятия'][i] - df['Дата договора'][i])


df['Количество дней договора'] = df['Количество дней договора'].apply(str)
df['Количество дней договора'] = df['Количество дней договора'].str.replace('NaT','0')
df['Количество дней договора'] = df['Количество дней договора'].str.replace('+','')
df['Количество дней договора'] = df['Количество дней договора'].str.replace('-','')
df['Количество дней договора'] = df['Количество дней договора'].str.replace(' days 00:00:00','')
df['Количество дней договора'] = df['Количество дней договора'].astype(int)

df['Дилер ЕСППО'] = df['Дилер ЕСППО'].apply(str)
df['Дилер ЕСППО'] = df['Дилер ЕСППО'].str.replace('не определен','0')

#показывает на 2 меньше
i = 0
for i in range(11718):
    if((df['Дата снятия'][i] - df['Дата договора'][i]) == max1):
        maxind = i
#print('index max:', maxind)

i = 0
for i in range(11718):
    if((df['Дата снятия'][i] - df['Дата договора'][i]) == min0):
        minind = i
#print('index max:', minind)

#Для цикла преобразования всех уникальных строк в уникальные числа
categorical_columns = df.select_dtypes(include='object').columns
encoder = preprocessing.LabelEncoder()

#del df['Unnamed: 0']
#del df['Unnamed: 0.1']
#del df['Дом']
#del df['Серийный номер приставки Sdp']
#del df['Дата возврата приставки']

      
#ЗАЕБАЛСЯ КРЧ ВСЁ СДЕЛЮ ВРУЧНУЮ ПОКА
#ЗАЕБАЛСЯ ПИСАТЬ ВРУЧНУЮ ПОПРОБУЮ СНОВА СДЕЛАТЬ КАК НАДО
#Для избавления от ошибки какой-то
le = LabelEncoder()
df = df.apply(lambda col: le.fit_transform(col.astype(str)), axis=0, result_type='expand')
col=0
for col in categorical_columns:
    #print(col)
    df[col] = df[col].fillna('missing value')
    df[col] = encoder.fit_transform(df[col])
    #print(df[col], 'max:', max(df[col]))


#Преобразование
#Приравнять всё к 1
#pd.options.mode.chained_assignment = None
#categorical_columns = df.select_dtypes(include='int').columns
col=0
i = 0
a = 0
#print(df.iterrows)
#columns вошаебная функция с помощю которой всё работает
for col in df.columns :
    df[col] = df[col].astype(float)
    maxcol = max(df[col])
    #print(col, maxcol)
    a = a + 1
    #print(a)
    for i in range(11718) :   
        if df[col][i] == 0 or df[col][i] == 'NaN' :
            df[col][i] = 1
        else : df[col][i] = df[col][i] / maxcol
    #print(col, maxcol)

forest = RandomForestClassifier()
X = df.values[:, 0:50]
Y = df.values[:, 4]
trainX, testX, trainY, testY = train_test_split( X, Y, test_size = 0.3)
forest.fit(trainX, trainY)
print('Accuracy: \n', forest.score(testX, testY))
pred = forest.predict(testX)

df.to_excel("aaa/output.xlsx")
