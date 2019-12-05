# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 18:20:03 2019

@author: Jinkai Zhang
"""

#import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer,MinMaxScaler
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression,Ridge,Lasso,LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from keras.models import Sequential
from keras.layers import Dense

#from sklearn.metrics import mean_squared_error

#read data
train=pd.read_csv('Desktop/Kaggle/train2.csv')
test=pd.read_csv('Desktop/Kaggle/test.csv')

#fill in the missing value and transfer back to dataframe
imp=Imputer(strategy="median")
train=pd.DataFrame(imp.fit_transform(train),columns=train.columns)
test=pd.DataFrame(imp.fit_transform(test),columns=test.columns)

#add more feature for intra day prediction
def add_feature(x):
    col_name=['Ret_%d'%(i) for i in range(1,121)]
    x['CRet_Mor']=x[col_name[1]]
    x['CRet_Noon']=x[col_name[90]]
    for i in range(2,90): x['CRet_Mor']=x['CRet_Mor']+x[col_name[i]]
    for i in range(91,120): x['CRet_Noon']=x['CRet_Noon']+x[col_name[i]]
    x['SD_Mor'],x['SD_Noon'],x['High_1'],x['High_2'],x['High_3'],\
    x['High_4'],x['High_5'],x['High_6']=0,0,0,0,0,0,0,0
    x['SD_Mor']=x.apply(lambda x:np.std(x[col_name[1:90]]),axis=1)
    x['SD_Noon']=x.apply(lambda x:np.std(x[col_name[90:121]]),axis=1)
    x['High_1']=x.apply(lambda x:sum(x[col_name[90:95]]),axis=1)
    print('good')
    x['High_2']=x.apply(lambda x:sum(x[col_name[95:100]]),axis=1)
    x['High_3']=x.apply(lambda x:sum(x[col_name[100:105]]),axis=1)
    x['High_4']=x.apply(lambda x:sum(x[col_name[105:110]]),axis=1)
    x['High_5']=x.apply(lambda x:sum(x[col_name[110:115]]),axis=1)
    x['High_6']=x.apply(lambda x:sum(x[col_name[115:120]]),axis=1)
    #drop some features to alleviate multicolinerity
    drop_num=[1,90,91,95,100,105,110,115]
    drop_name=[x for x in col_name if col_name.index(x) in drop_num]
    x=x.drop(columns=drop_name)
    return x

train2=add_feature(train)
test2=add_feature(test)

#write csv
train2.to_csv('Desktop/Kaggle/train_2.csv')
test2.to_csv('Desktop/Kaggle/test_3.csv')

train2 = pd.read_csv('Desktop/Kaggle/train_2.csv')
test2 = pd.read_csv('Desktop/Kaggle/test_3.csv')

train2=train2.iloc[:,1:]
test2=test2.iloc[:,1:]


#Table 1 and 2 refers to training and test table respectively
T1_feature=train2.iloc[:,1:139].merge(train2.iloc[:,203:213],\
                        left_index = True,right_index = True,how = 'outer')
T1_value=train2.iloc[:,139:201]
T1_intra_weight=train2.loc[:,'Weight_Intraday']
T1_daily_weight=train2.loc[:,'Weight_Daily']

scl=MinMaxScaler()
scl.fit(T2_feature)
T2_feature=pd.DataFrame(scl.transform(T2_feature),columns=T2_feature.columns)

#scale the feature using min-max method
scl=MinMaxScaler()
scl.fit(T1_feature)
T1_feature=pd.DataFrame(scl.transform(T1_feature),columns=T1_feature.columns)

#set 90% data as traning data
cut=round(0.9*len(T1_feature))

T1_train=T1_feature.iloc[:cut,:]
T1_intra_lable=T1_value.iloc[:cut,:T1_value.shape[1]-2]
T1_daily_lable=T1_value.iloc[:cut,T1_value.shape[1]-2:]

T1_test=T1_feature.iloc[cut:,:]
T1_intra_true=T1_value.iloc[cut:,:T1_value.shape[1]-2]
T1_daily_true=T1_value.iloc[cut:,T1_value.shape[1]-2:]

#define MAE performance function
def Weighted_averege_MAE(y_predict,y_true,weight):
    MAE=np.mean(np.mean(np.abs(y_predict-y_true), axis=1)* weight)
    return MAE

#linear regression
lm_intra=LinearRegression()
lm_intra.fit(T1_train, T1_intra_lable, T1_intra_weight[:cut])
lm_intra_MAE=Weighted_averege_MAE(lm_intra.predict(T1_test),\
                        T1_intra_true.as_matrix(),T1_intra_weight[cut:])

lm_daily=LinearRegression()
lm_daily.fit(T1_train, T1_daily_lable, T1_daily_weight[:cut])
lm_daily_MAE=Weighted_averege_MAE(lm_daily.predict(T1_test),\
                        T1_daily_true.as_matrix(),T1_daily_weight[cut:])


#ridge regression
#The performance of alpha=1 better than alpha=0.5
rm_intra=Ridge(alpha=1,normalize=True)
rm_intra.fit(T1_train, T1_intra_lable, T1_intra_weight[:cut])
rm_intra_MAE=Weighted_averege_MAE(rm_intra.predict(T1_test),\
                                  T1_intra_true.as_matrix(),T1_intra_weight[cut:])


rm_daily=Ridge(alpha=1,normalize=True)
rm_daily.fit(T1_train, T1_daily_lable, T1_daily_weight[:cut])
rm_daily_MAE=Weighted_averege_MAE(rm_daily.predict(T1_test),\
                                  T1_daily_true.as_matrix(),T1_daily_weight[cut:])

#LASSO regression
lasso_intra=Lasso(normalize=True,alpha=0.00001)
lasso_intra.fit(T1_train, T1_intra_lable)
lasso_intra_MAE=Weighted_averege_MAE(lasso_intra.predict(T1_test),\
                                  T1_intra_true.as_matrix(),T1_intra_weight[cut:])

lasso_daily=Lasso(normalize=True,alpha=0.00001)
lasso_daily.fit(T1_train, T1_daily_lable)
lasso_daily_MAE=Weighted_averege_MAE(lasso_daily.predict(T1_test),\
                                  T1_daily_true.as_matrix(),T1_daily_weight[cut:])
'''
#Logistic regression
logic_intra=LogisticRegression()
logic_intra.fit(T1_train, T1_intra_lable, T1_intra_weight[:cut])
logic_intra_MAE=Weighted_averege_MAE(logic_intra.predict(T1_test),\
                                  T1_intra_true.as_matrix(),T1_intra_weight[cut:])

logic_daily=LogisticRegression()
logic_daily.fit(T1_train, T1_daily_lable)
logic_daily_MAE=Weighted_averege_MAE(logic_daily.predict(T1_test),\
                                  T1_daily_true.as_matrix(),T1_daily_weight[cut:])
'''
#Random forest
#Parameter seleceted from experiment; I finally choose depth=25 (30) for intraday (daily)
def Best_RF_Model_Intra(T1_train,T1_intra_lable,T1_intra_weight,\
                        cut,T1_test,T1_intra_true):
    min_samples_split=[5]
    max_depth=[15,20,25,30]
    n_estimators=[10]
    rf_intra_MAE=100000
    Intra_para=[]
    criterion=['mse']
    for split in min_samples_split:
        for depth in max_depth:
            for num in n_estimators:
                for cri in criterion:
                    rf_intra=RandomForestRegressor(criterion=cri,\
                               min_samples_split=split,max_depth=depth,n_estimators=num)
                    rf_intra.fit(T1_train, T1_intra_lable, T1_intra_weight[:cut])
                    MAE=Weighted_averege_MAE(rf_intra.predict(T1_test),\
                                  T1_intra_true.as_matrix(),T1_intra_weight[cut:])
                    if MAE<=rf_intra_MAE:
                        rf_intra_MAE=MAE
                        Intra_para=[split,depth,num,cri]
                        print(rf_intra_MAE)
                        print(Intra_para)
    return  [rf_intra_MAE, Intra_para]         
                
def Best_RF_Model_Daily(T1_train,T1_daily_lable,T1_daily_weight,cut,T1_test,\
                        T1_daily_true):
    min_samples_split=[3,5,7]
    max_depth=[10,15,20,30,50]
    n_estimators=[10]
    rf_Daily_MAE=100000
    Daily_para=[]
    criterion=['mse']
    for split in min_samples_split:
        for depth in max_depth:
            for num in n_estimators:
                for cri in criterion:
                    rf_daily=RandomForestRegressor(criterion=cri, \
                        min_samples_split=split,max_depth=depth,n_estimators=num)
                    rf_daily.fit(T1_train, T1_daily_lable, T1_daily_weight[:cut])
                    MAE=Weighted_averege_MAE(rf_daily.predict(T1_test),\
                                  T1_daily_true.as_matrix(),T1_daily_weight[cut:])
                    if MAE<=rf_Daily_MAE:
                        rf_Daily_MAE=MAE
                        Daily_para=[split,depth,num,cri]
                        print(rf_Daily_MAE)
                        print(Daily_para)
    return  [rf_Daily_MAE, Daily_para]      

Best_Ramdom_Intra=Best_RF_Model_Intra(T1_train,T1_intra_lable,T1_intra_weight,\
                        cut,T1_test,T1_intra_true)
Best_Ramdom_Daily=Best_RF_Model_Daily(T1_train,T1_daily_lable,T1_daily_weight,cut,T1_test,\
                        T1_daily_true)

    

rf_intra=RandomForestRegressor(criterion='mse', \
                               min_samples_split=5,max_depth=25)
rf_intra.fit(T1_train, T1_intra_lable, T1_intra_weight[:cut])
rf_intra_MAE=Weighted_averege_MAE(rf_intra.predict(T1_test),\
                                  T1_intra_true.as_matrix(),T1_intra_weight[cut:])

rf_daily=RandomForestRegressor(criterion='mse', \
                               min_samples_split=5,max_depth=30)
rf_daily.fit(T1_train, T1_daily_lable, T1_daily_weight[:cut])
rf_daily_MAE=Weighted_averege_MAE(rf_daily.predict(T1_test),\
                                  T1_daily_true.as_matrix(),T1_daily_weight[cut:])

#svr model

def SVR_Intra_MAE(T1_train,T1_intra_lable,T1_test,\
                    T1_intra_true,T1_intra_weight,cut):
    col_name=['Ret_%d'%(i) for i in range(1,61)]
    Predic=pd.DataFrame()
    for i in range(60):
        svr_intra=SVR(kernel ='linear',C=1000,gamma=1e-3,epsilon=0.05)        
        T1_train_scale=MinMaxScaler(feature_range=(-1, 1))
        T1_train_scale_fit=T1_train_scale.fit(T1_train)
        T1_train_scale_fit=T1_train_scale.transform(T1_train)
        svr_intra.fit(T1_train_scale_fit, T1_intra_lable.iloc[:,i])
        T1_test_scale=MinMaxScaler()
        T1_test_scale_fit=T1_test_scale.fit(T1_test)
        T1_test_scale_fit=T1_test_scale.transform(T1_test)
        svr_intra.predict(T1_test_scale_fit)
        Predic[col_name[i]]=svr_intra.predict(T1_test_scale_fit)
        print (i)
    svr_intra_MAE=Weighted_averege_MAE(Predic.as_matrix(),T1_intra_true.as_matrix(),\
                                       T1_intra_weight[cut:])    
    return svr_intra_MAE

        
svr_intra_MAE=SVR_Intra_MAE(T1_train,T1_intra_lable,T1_test,\
                    T1_intra_true,T1_intra_weight,cut)    


def SVR_Daily_MAE(T1_train,T1_daily_lable,T1_test,\
                    T1_daily_true,T1_daily_weight,cut):
    col_name=['Ret_%d'%(i) for i in range(1,61)]
    Predic=pd.DataFrame()
    for i in range(2):
        svr_daily=SVR(kernel ='sigmoid',C=1000,gamma=1e-2,epsilon=0.001)        
        T1_train_scale=MinMaxScaler(feature_range=(-1, 1))
        T1_train_scale_fit=T1_train_scale.fit(T1_train)
        T1_train_scale_fit=T1_train_scale.transform(T1_train)
        svr_daily.fit(T1_train_scale_fit, T1_intra_lable.iloc[:,i])
        T1_test_scale=MinMaxScaler()
        T1_test_scale_fit=T1_test_scale.fit(T1_test)
        T1_test_scale_fit=T1_test_scale.transform(T1_test)
        svr_daily.predict(T1_test_scale_fit)
        Predic[col_name[i]]=svr_daily.predict(T1_test_scale_fit)
        print (i)
    svr_daily_MAE=Weighted_averege_MAE(Predic.as_matrix(),T1_daily_true.as_matrix(),\
                                       T1_daily_weight[cut:])    
    return svr_daily_MAE

        
svr_daily_MAE=SVR_Daily_MAE(T1_train,T1_daily_lable,T1_test,\
                    T1_daily_true,T1_daily_weight,cut)    
svr_daily_MAE


#Deep learning

def Deep_Intra_model(T1_train,T1_intra_lable,T1_test,\
                    T1_intra_true,T1_intra_weight,cut):
    col_name=['Ret_%d'%(i) for i in range(1,61)]
    Predic=pd.DataFrame()
    for i in range(60):
        Deep_Intra=Sequential()
        Deep_Intra.add(Dense(12, input_dim=148, kernel_initializer='normal'))
        Deep_Intra.add(Dense(8, activation='relu'))
        Deep_Intra.add(Dense(1, activation='linear'))
        Deep_Intra.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])
        Deep_Intra.fit(T1_train,T1_intra_lable.iloc[:,i],epochs=10, batch_size=50)
        Predic[col_name[i]]=np.squeeze(np.asarray(Deep_Intra.predict(T1_test)))
        print (i)
    Deep_intra_MAE=Weighted_averege_MAE(Predic.as_matrix(),T1_intra_true.as_matrix(),\
                                       T1_intra_weight[cut:]) 
    return Deep_intra_MAE
    
Deep_intra_MAE=Deep_Intra_model(T1_train,T1_intra_lable,T1_test,\
                    T1_intra_true,T1_intra_weight,cut)


def Deep_Daily_model(T1_train,T1_daily_lable,T1_test,\
                    T1_daily_true,T1_daily_weight,cut):
    col_name=['Day_1','Day_2']
    Predic=pd.DataFrame()
    for i in range(2):
        Deep_Daily=Sequential()
        Deep_Daily.add(Dense(12, input_dim=148, kernel_initializer='normal'))
        Deep_Daily.add(Dense(8, activation='relu'))
        Deep_Daily.add(Dense(1, activation='linear'))
        Deep_Daily.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])
        Deep_Daily.fit(T1_train,T1_daily_lable.iloc[:,i],epochs=10, batch_size=50)
        Predic[col_name[i]]=np.squeeze(np.asarray(Deep_Daily.predict(T1_test)))
    Deep_daily_MAE=Weighted_averege_MAE(Predic.as_matrix(),T1_daily_true.as_matrix(),\
                                       T1_daily_weight[cut:])
    return Deep_daily_MAE

Deep_daily_MAE=Deep_Daily_model(T1_train,T1_daily_lable,T1_test,\
                    T1_daily_true,T1_daily_weight,cut)    
    

#consider the combination of Lasso and Random forest

Predict_rf=rf_intra.predict(T1_test)
Predict_lasso=lasso_intra.predict(T1_test)

def Best_Model_Weight(T1_test,T1_intra_true,T1_intra_weight,cut,Predict_rf,Predict_lasso):
    weight_rf=np.arange(0.0, 1.01, 0.01)
    MAE=100000
    Best_Weight_rf=0
    for w in weight_rf:
        weight_lasso=1-w
        Predict_comb=w*Predict_rf+weight_lasso*Predict_lasso
        MAE_Both=Weighted_averege_MAE(Predict_comb,\
                                  T1_intra_true.as_matrix(),T1_intra_weight[cut:])
        if MAE_Both<MAE:
            MAE=MAE_Both
            print (MAE_Both)
            Best_Weight_rf=w
    return  Best_Weight_rf
Best_rf=Best_Model_Weight(T1_test,T1_intra_true,T1_intra_weight,cut,Predict_rf,Predict_lasso)


# weight of ramdon forest for daily return is 1
'''
def Best_Model_Weight(T1_test,T1_daily_true,T1_daily_weight,cut,Predict_rf,Predict_lasso):
    weight_rf=np.arange(0.0, 1.01, 0.01)
    MAE=100000
    Best_Weight_rf=0

    for w in weight_rf:
        weight_lasso=1-w
        Predict_comb=w*Predict_rf+weight_lasso*Predict_lasso
        MAE_Both=Weighted_averege_MAE(Predict_comb,\
                                  T1_daily_true.as_matrix(),T1_daily_weight[cut:])
        if MAE_Both<MAE:
            print (MAE_Both)
            Best_Weight_rf=w
    return  Best_Weight_rf
 '''     


#The weight for random forest in intra day is 0.87 and in daily is 1
T2_feature=test2.iloc[:,1:150]
T1_value_intra=T1_value.iloc[:,:60]
T1_value_daily=T1_value.iloc[:,60:]

scl=MinMaxScaler()
scl.fit(T2_feature)
T2_feature=pd.DataFrame(scl.transform(T2_feature),columns=T2_feature.columns)

#Intra day prediction 0.87 random forest (depth=25) + 0.13 Lasso
lasso_intra_T2=Lasso(normalize=True,alpha=0.00001)
lasso_intra_T2.fit(T1_feature,T1_value_intra)
Pre_lasso_intra=lasso_intra_T2.predict(T2_feature)
rf_intra_T2=RandomForestRegressor(criterion='mse', \
                               min_samples_split=5,max_depth=25)
rf_intra_T2.fit(T1_feature,T1_value_intra,T1_intra_weight)
Pre_rf_intra=rf_intra_T2.predict(T2_feature)
Pre_intra=0.87*Pre_rf_intra+0.13*Pre_lasso_intra

#daily prediction using random forest (depth=30)
rf_daily_T2=RandomForestRegressor(criterion='mse', \
                               min_samples_split=5,max_depth=30)
rf_daily_T2.fit(T1_feature,T1_value_daily,T1_daily_weight)
Pre_daily=rf_daily_T2.predict(T2_feature)
Pre_all=np.concatenate((Pre_intra,Pre_daily),axis=1)


'''
Submission=pd.DataFrame(columns = ["Id", "Predicted"])

for i in range(120000):
    for j in range(62):
        num=i+1
        day=j+1
        k=i*62+j
        value_id=str(num)+'_'+str(day)
        value_ret=Pre_all[i][j]
        Submission=Submission.append(pd.DataFrame({"Id":[value_id],"Predicted":[value_ret]}))
        if k % 1000==0: print (k)   
stock_id=[i for i in range(1,120001)]       
'''
#submission  
Submission=pd.DataFrame(columns = ["Id", "Predicted"])
col_id=[]
col_ret=[]
for i in range(120000):
    col_ret.extend(Pre_all[i])
for i in range(120000):
    for j in range(62):
        num=i+1
        day=j+1
        k=i*62+j
        col_id.append(str(num)+'_'+str(day))        
Submission=Submission.append(pd.DataFrame({"Id":col_id,"Predicted":col_ret}))        
Submission.to_csv('Desktop/Kaggle/submission_all.csv',index = False)
'''
Submission.iloc[:1000000,:].to_csv('Desktop/Kaggle/submission_1.csv',index = False)
Submission.iloc[1000000:2000000,:].to_csv('Desktop/Kaggle/submission_2.csv',index = False)
Submission.iloc[2000000:3000000,:].to_csv('Desktop/Kaggle/submission_3.csv',index = False)
Submission.iloc[3000000:4000000,:].to_csv('Desktop/Kaggle/submission_4.csv',index = False)
Submission.iloc[4000000:5000000,:].to_csv('Desktop/Kaggle/submission_5.csv',index = False)
Submission.iloc[5000000:6000000,:].to_csv('Desktop/Kaggle/submission_6.csv',index = False)
Submission.iloc[6000000:7000000,:].to_csv('Desktop/Kaggle/submission_7.csv',index = False)
Submission.iloc[7000000:7440000,:].to_csv('Desktop/Kaggle/submission_8.csv',index = False)
'''

