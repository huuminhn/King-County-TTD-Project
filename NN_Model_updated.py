#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection  import train_test_split
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC 
from sklearn.neural_network import MLPClassifier 
from scipy import stats
from sklearn.impute import KNNImputer
import numpy as np
import seaborn as sns
# from fancyimpute import KNN
import statsmodels.api as sm
from sklearn import tree
import graphviz #optional –needed to render a tree model into graph
import warnings
warnings.filterwarnings('ignore')


# In[5]:

DFA_train = pd.read_excel('DFAtrain_update.xlsx')

DFA_test = pd.read_excel('DFA_Test_2.xlsx')

DFA_test['Violent Level'] = np.where(DFA_test['Violent'] == 0, 'Non-violent Charge', 'Violent Charge')

DFA_final = pd.concat([DFA_train, DFA_test], ignore_index=True, sort=False)
DFA_final.dropna(subset=['Seriousness'], inplace=True)

#DFA_final = DFA_final.drop(['Unnamed: 0'], axis = 1)


# In[6]:


# Create dummies for train data
cat_vars=['AgeGroup','Gender','Venue_Dummies', 'Police_Dummies','Highest Class',
          'Violent Level','Previous DFA', 'hearingsDummies', 'settingDummies', 'Old Record']
for var in cat_vars:
    cat_list='var'+'_'+var
    cat_list = pd.get_dummies(DFA_final[var], drop_first = True, prefix=var)
    data1=DFA_final.join(cat_list)
    DFA_final=data1


# In[7]:


DFA_final = DFA_final.drop(columns = ['File Number','Defendant ID','DOB Anon','Referral Date','Case Types',
         'Event Code','Event Enter Date','Police Agency','Anon LE Number',
         'Venue','Age','Case No 1','Case No 2','Case No 3','Case No 4',
         'Case No 5','Case No 6','Violent','Previous DFA','Gender', 'AgeGroup',
         'Venue_Dummies','Police_Dummies', 'Highest Class', 'Violent Level',
         'preDFA', 'hearingsDummies', 'settingDummies', ''
         'Old Record','Others'], axis = 1)

DFA_final.dtypes


# In[10]:


y_DFA = DFA_final['DFA'].astype('category').cat.codes

DFA_final['intercept'] = 1.0

X_DFA = DFA_final.drop(columns = ['DFA','FailureToAppear','DFAAfter','DFABefore',
                                 'AgeGroup_<19','eDiscovery','AdultMisdemeanor',
                                 'AgeGroup_30-39','JuvenileFelony','JuvenileMisdemeanor',
                                 'Gender_Unknown','Venue_Dummies_Other Venues',
                                 'Venue_Dummies_Seattle Venue','Police_Dummies_Agent7'], axis=1)
X_DFA3 = DFA_final.drop(columns = ['DFA','FailureToAppear','DFAAfter','DFABefore'], axis =1)

X_DFA3.count()
X_DFA3 = X_DFA3[['Seriousness','Domestic violence','Violent Level_Violent Charge',
                 'AdultFelony','Venue_Dummies_Seattle Venue',
                 'Highest Class_M','Previous DFA_1']]

# Validation: Holdout method
X_trainDFA3, X_testDFA3, y_trainDFA3, y_testDFA3 = train_test_split(X_DFA3, y_DFA,
                                                test_size=0.20,random_state=109) 


# ### All varaiables model

# #### Identity Activation

# In[32]:


scaler = StandardScaler()  
scaler.fit(X_trainDFA3)
X_trainDFA3 = scaler.transform(X_trainDFA3)  
X_testDFA3 = scaler.transform(X_testDFA3)

accuracy1 = []
precision1 = []
recall1 = []
f11 = []
xsize = X_trainDFA3.shape
step = 1
xaxis = range(1,10,step)

actFunc = ['identity', 'logistic', 'tanh', 'relu']
for x in xaxis:
    ## Multi-layer perceptron classification - one hidden layer of 3 neurons
    mlp = MLPClassifier(hidden_layer_sizes=(x), max_iter=1000,random_state = 109, activation = 'identity')  
    mlp.fit(X_trainDFA3, y_trainDFA3)  

    ## predict test set 
    y_predDFA3 = mlp.predict(X_testDFA3)

    accuracy1.append(metrics.accuracy_score(y_testDFA3,y_predDFA3)*100)
    precision1.append(metrics.precision_score(y_testDFA3,y_predDFA3)*100)
    recall1.append(metrics.recall_score(y_testDFA3,y_predDFA3)*100)
    f11.append(metrics.f1_score(y_testDFA3,y_predDFA3)*100)

maxAccuracyIndex = accuracy1.index(max(accuracy1))
maxPrecisionIndex = precision1.index(max(precision1))
maxRecallIndex = recall1.index(max(recall1))
maxF1Index = f11.index(max(f11))

print("max min_sample_leaf accuracy: " + str(round(max(accuracy1),2)) + " index " + str(maxAccuracyIndex*step+1))
print("max min_sample_leaf precision: " + str(round(max(precision1),2)) + " index " + str(maxPrecisionIndex*step+1))
print("max min_sample_leaf recall: " + str(round(max(recall1),2)) + " index " + str(maxRecallIndex*step+1))
print("max min_sample_leaf f1: " + str(round(max(f11),2)) + " index " + str(maxF1Index*step+1))


print("At max accuracy index " + str(maxAccuracyIndex*step+1) + 
      "\n Accuracy:  " + str(round(accuracy1[maxAccuracyIndex],2)) + "%"
      "\n Precision:  " + str(round(precision1[maxAccuracyIndex],2)) + "%"
      "\n Recall:  " + str(round(recall1[maxAccuracyIndex],2))+ "%" +
      "\n F1:  " + str(round(f11[maxAccuracyIndex],2))+ "%")

print("At max Precision index " + str(maxPrecisionIndex*step+1) + 
      "\n Accuracy:  " + str(round(accuracy1[maxPrecisionIndex],2)) + "%"
      "\n Precision:  " + str(round(precision1[maxPrecisionIndex],2)) + "%"
      "\n RecallFf1:  " + str(round(recall1[maxPrecisionIndex],2))+ "%"+
      "\n F1:  " + str(round(f11[maxPrecisionIndex],2))+ "%") 

print("At max Recall index " + str(maxRecallIndex*step+1) + 
      "\n Accuracy:  " + str(round(accuracy1[maxRecallIndex],2)) + "%"
      "\n Precision:  " + str(round(precision1[maxRecallIndex],2)) + "%"
      "\n Recall:  " + str(round(recall1[maxRecallIndex],2))+ "%"+
      "\n F1:  " + str(round(f11[maxRecallIndex],2))+ "%")    

print("At max f1 index " + str(maxF1Index*step+1) + 
      "\n Accuracy:  " + str(round(accuracy1[maxF1Index],2)) + "%"
      "\n Precision:  " + str(round(precision1[maxF1Index],2)) + "%"
      "\n Recall:  " + str(round(recall1[maxF1Index],2))+ "%"+
      "\n F1:  " + str(round(f11[maxF1Index],2))+ "%")    
    


# #### Logistic Activation

# In[33]:


accuracy2 = []
precision2 = []
recall2 = []
f12 = []
xsize = X_trainDFA3.shape
step = 1
xaxis = range(1,10,step)

actFunc = ['identity', 'logistic', 'tanh', 'relu']
for x in xaxis:
    ## Multi-layer perceptron classification - one hidden layer of 3 neurons
    mlp = MLPClassifier(hidden_layer_sizes=(x), max_iter=1000,random_state = 109, activation = 'logistic')  
    mlp.fit(X_trainDFA3, y_trainDFA3)  

    ## predict test set 
    y_predDFA3 = mlp.predict(X_testDFA3)

    accuracy2.append(metrics.accuracy_score(y_testDFA3,y_predDFA3)*100)
    precision2.append(metrics.precision_score(y_testDFA3,y_predDFA3)*100)
    recall2.append(metrics.recall_score(y_testDFA3,y_predDFA3)*100)
    f12.append(metrics.f1_score(y_testDFA3,y_predDFA3)*100)

maxAccuracyIndex = accuracy2.index(max(accuracy2))
maxPrecisionIndex = precision2.index(max(precision2))
maxRecallIndex = recall2.index(max(recall2))
maxF1Index = f12.index(max(f12))

print("max min_sample_leaf accuracy: " + str(round(max(accuracy2),2)) + " index " + str(maxAccuracyIndex*step+1))
print("max min_sample_leaf precision: " + str(round(max(precision2),2)) + " index " + str(maxPrecisionIndex*step+1))
print("max min_sample_leaf recall: " + str(round(max(recall2),2)) + " index " + str(maxRecallIndex*step+1))
print("max min_sample_leaf f1: " + str(round(max(f12),2)) + " index " + str(maxF1Index*step+1))


print("At max accuracy index " + str(maxAccuracyIndex*step+1) + 
      "\n Accuracy:  " + str(round(accuracy2[maxAccuracyIndex],2)) + "%"
      "\n Precision:  " + str(round(precision2[maxAccuracyIndex],2)) + "%"
      "\n Recall:  " + str(round(recall2[maxAccuracyIndex],2))+ "%" +
      "\n Recall:  " + str(round(f12[maxAccuracyIndex],2))+ "%")

print("At max Precision index " + str(maxPrecisionIndex*step+1) + 
      "\n Accuracy:  " + str(round(accuracy2[maxPrecisionIndex],2)) + "%"
      "\n Precision:  " + str(round(precision2[maxPrecisionIndex],2)) + "%"
      "\n Recall:  " + str(round(recall2[maxPrecisionIndex],2))+ "%"+
      "\n Recall:  " + str(round(f12[maxPrecisionIndex],2))+ "%") 

print("At max Recall index " + str(maxRecallIndex*step+1) + 
      "\n Accuracy:  " + str(round(accuracy2[maxRecallIndex],2)) + "%"
      "\n Precision:  " + str(round(precision2[maxRecallIndex],2)) + "%"
      "\n Recall:  " + str(round(recall2[maxRecallIndex],2))+ "%"+
      "\n Recall:  " + str(round(f12[maxRecallIndex],2))+ "%")    

print("At max f1 index " + str(maxF1Index*step+1) + 
      "\n Accuracy:  " + str(round(accuracy2[maxF1Index],2)) + "%"
      "\n Precision:  " + str(round(precision2[maxF1Index],2)) + "%"
      "\n Recall:  " + str(round(recall2[maxF1Index],2))+ "%"+
      "\n Recall:  " + str(round(f12[maxF1Index],2))+ "%")   


# #### tanh Activation

# In[60]:


accuracy3 = []
precision3 = []
recall3 = []
f13 = []
xsize = X_trainDFA3.shape
step = 1
xaxis = range(1,10,step)

actFunc = ['identity', 'logistic', 'tanh', 'relu']
for x in xaxis:
    ## Multi-layer perceptron classification - one hidden layer of 3 neurons
    mlp = MLPClassifier(hidden_layer_sizes=(x), max_iter=1000,random_state = 109, activation = 'tanh')  
    mlp.fit(X_trainDFA3, y_trainDFA3)  

    ## predict test set 
    y_predDFA3 = mlp.predict(X_testDFA3)

    accuracy3.append(metrics.accuracy_score(y_testDFA3,y_predDFA3)*100)
    precision3.append(metrics.precision_score(y_testDFA3,y_predDFA3)*100)
    recall3.append(metrics.recall_score(y_testDFA3,y_predDFA3)*100)
    f13.append(metrics.f1_score(y_testDFA3,y_predDFA3)*100)

maxAccuracyIndex = accuracy3.index(max(accuracy3))
maxPrecisionIndex = precision3.index(max(precision3))
maxRecallIndex = recall3.index(max(recall3))
maxF1Index = f13.index(max(f13))

print("max min_sample_leaf accuracy: " + str(round(max(accuracy3),2)) + " index " + str(maxAccuracyIndex*step+1))
print("max min_sample_leaf precision: " + str(round(max(precision3),2)) + " index " + str(maxPrecisionIndex*step+1))
print("max min_sample_leaf recall: " + str(round(max(recall3),2)) + " index " + str(maxRecallIndex*step+1))
print("max min_sample_leaf f1: " + str(round(max(f13),2)) + " index " + str(maxF1Index*step+1))


print("At max accuracy index " + str(maxAccuracyIndex*step+1) + 
      "\n Accuracy:  " + str(round(accuracy3[maxAccuracyIndex],2)) + "%"
      "\n Precision:  " + str(round(precision3[maxAccuracyIndex],2)) + "%"
      "\n Recall:  " + str(round(recall3[maxAccuracyIndex],2))+ "%" +
      "\n Recall:  " + str(round(f13[maxAccuracyIndex],2))+ "%")

print("At max Precision index " + str(maxPrecisionIndex*step+1) + 
      "\n Accuracy:  " + str(round(accuracy3[maxPrecisionIndex],2)) + "%"
      "\n Precision:  " + str(round(precision3[maxPrecisionIndex],2)) + "%"
      "\n Recall:  " + str(round(recall3[maxPrecisionIndex],2))+ "%"+
      "\n Recall:  " + str(round(f13[maxPrecisionIndex],2))+ "%") 

print("At max Recall index " + str(maxRecallIndex*step+1) + 
      "\n Accuracy:  " + str(round(accuracy3[maxRecallIndex],2)) + "%"
      "\n Precision:  " + str(round(precision3[maxRecallIndex],2)) + "%"
      "\n Recall:  " + str(round(recall3[maxRecallIndex],2))+ "%"+
      "\n Recall:  " + str(round(f13[maxRecallIndex],2))+ "%")    

print("At max f1 index " + str(maxF1Index*step+1) + 
      "\n Accuracy:  " + str(round(accuracy3[maxF1Index],2)) + "%"
      "\n Precision:  " + str(round(precision3[maxF1Index],2)) + "%"
      "\n Recall:  " + str(round(recall3[maxF1Index],2))+ "%"+
      "\n Recall:  " + str(round(f13[maxF1Index],2))+ "%")   
    


# #### relu Activation

# In[59]:


accuracy4 = []
precision4 = []
recall4 = []
f14 = []
xsize = X_trainDFA3.shape
step = 1
xaxis = range(1,10,step)

actFunc = ['identity', 'logistic', 'tanh', 'relu']
for x in xaxis:
    ## Multi-layer perceptron classification - one hidden layer of 3 neurons
    mlp = MLPClassifier(hidden_layer_sizes=(x), max_iter=1000,random_state = 109, activation = 'relu')  
    mlp.fit(X_trainDFA3, y_trainDFA3)  

    ## predict test set 
    y_predDFA3 = mlp.predict(X_testDFA3)

    accuracy4.append(metrics.accuracy_score(y_testDFA3,y_predDFA3)*100)
    precision4.append(metrics.precision_score(y_testDFA3,y_predDFA3)*100)
    recall4.append(metrics.recall_score(y_testDFA3,y_predDFA3)*100)
    f14.append(metrics.f1_score(y_testDFA3,y_predDFA3)*100)

maxAccuracyIndex = accuracy4.index(max(accuracy4))
maxPrecisionIndex = precision4.index(max(precision4))
maxRecallIndex = recall4.index(max(recall4))
maxF1Index = f14.index(max(f14))

print("max min_sample_leaf accuracy: " + str(round(max(accuracy4),2)) + " index " + str(maxAccuracyIndex*step+1))
print("max min_sample_leaf precision: " + str(round(max(precision4),2)) + " index " + str(maxPrecisionIndex*step+1))
print("max min_sample_leaf recall: " + str(round(max(recall4),2)) + " index " + str(maxRecallIndex*step+1))
print("max min_sample_leaf f1: " + str(round(max(f14),2)) + " index " + str(maxF1Index*step+1))


print("At max accuracy index " + str(maxAccuracyIndex*step+1) + 
      "\n Accuracy:  " + str(round(accuracy4[maxAccuracyIndex],2)) + "%"
      "\n Precision:  " + str(round(precision4[maxAccuracyIndex],2)) + "%"
      "\n Recall:  " + str(round(recall4[maxAccuracyIndex],2))+ "%" +
      "\n Recall:  " + str(round(f14[maxAccuracyIndex],2))+ "%")

print("At max Precision index " + str(maxPrecisionIndex*step+1) + 
      "\n Accuracy:  " + str(round(accuracy4[maxPrecisionIndex],2)) + "%"
      "\n Precision:  " + str(round(precision4[maxPrecisionIndex],2)) + "%"
      "\n Recall:  " + str(round(recall4[maxPrecisionIndex],2))+ "%"+
      "\n Recall:  " + str(round(f14[maxPrecisionIndex],2))+ "%") 

print("At max Recall index " + str(maxRecallIndex*step+1) + 
      "\n Accuracy:  " + str(round(accuracy4[maxRecallIndex],2)) + "%"
      "\n Precision:  " + str(round(precision4[maxRecallIndex],2)) + "%"
      "\n Recall:  " + str(round(recall4[maxRecallIndex],2))+ "%"+
      "\n Recall:  " + str(round(f14[maxRecallIndex],2))+ "%")    

print("At max f1 index " + str(maxF1Index*step+1) + 
      "\n Accuracy:  " + str(round(accuracy4[maxF1Index],2)) + "%"
      "\n Precision:  " + str(round(precision4[maxF1Index],2)) + "%"
      "\n Recall:  " + str(round(recall4[maxF1Index],2))+ "%"+
      "\n Recall:  " + str(round(f14[maxF1Index],2))+ "%")   


# In[74]:


fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2,figsize=(20,20))
ax1.plot( xaxis, accuracy1,   color='skyblue', linewidth=2, label = 'accuracy')
ax1.plot( xaxis, precision1,   color='green', linewidth=2, label = 'precision')
ax1.plot( xaxis, recall1,   color='red', linewidth=2,  label="recall")
ax1.plot( xaxis, f11,   color='purple', linewidth=2,  label="recall")
ax1.set_title('Identity Activation Function')
ax1.set_xlabel('Hidden Layers')
ax1.set_ylabel('Percentage')
ax1.set_yticks(np.arange(0, 100, 5.0))
ax1.set_xticks(np.arange(min(xaxis)-1, max(xaxis)+1, 1))


ax2.plot( xaxis, accuracy2,   color='skyblue', linewidth=2, label = 'accuracy')
ax2.plot( xaxis, precision2,   color='green', linewidth=2, label = 'precision')
ax2.plot( xaxis, recall2,   color='red', linewidth=2,  label="recall")
ax2.plot( xaxis, f12,   color='purple', linewidth=2,  label="recall")
ax2.set_title('Logistic Activation Function')
ax2.set_xlabel('Hidden Layers')
ax2.set_ylabel('Percentage')
ax2.set_yticks(np.arange(0, 100, 5.0))
ax2.set_xticks(np.arange(min(xaxis)-1, max(xaxis)+1, 1))

ax3.plot( xaxis, accuracy3,   color='skyblue', linewidth=2, label = 'accuracy')
ax3.plot( xaxis, precision3,   color='green', linewidth=2, label = 'precision')
ax3.plot( xaxis, recall3,   color='red', linewidth=2,  label="recall")
ax3.plot( xaxis, f13,   color='purple', linewidth=2,  label="recall")
ax3.set_title('TanH Activation Function')
ax3.set_xlabel('Hidden Layers')
ax3.set_ylabel('Percentage')
ax3.set_yticks(np.arange(0, 100, 5.0))
ax3.set_xticks(np.arange(min(xaxis)-1, max(xaxis)+1, 1))

ax4.plot( xaxis, accuracy4,   color='skyblue', linewidth=2, label = 'accuracy')
ax4.plot( xaxis, precision4,   color='green', linewidth=2, label = 'precision')
ax4.plot( xaxis, recall4,   color='red', linewidth=2,  label="recall")
ax4.plot( xaxis, f14,   color='purple', linewidth=2,  label="recall")
ax4.set_title('ReLu Activation Function')
ax4.set_xlabel('Hidden Layers')
ax4.set_ylabel('Percentage')
ax4.set_yticks(np.arange(0, 100, 5.0))
ax4.set_xticks(np.arange(min(xaxis)-1, max(xaxis)+1, 1))

plt.suptitle("All Variable NN", fontsize=30)


# #### New Model #2

# In[ ]:





# In[62]:


DFA_final = DFA_final.drop(columns = ['File Number','Defendant ID','DOB Anon','Referral Date','Case Types',
         'Event Code','Event Enter Date','Police Agency','Anon LE Number',
         'Venue','Age','Case No 1','Case No 2','Case No 3','Case No 4',
         'Case No 5','Case No 6','Violent','Previous DFA','Gender', 'AgeGroup',
         'Venue_Dummies','Police_Dummies', 'Highest Class', 'Violent Level',
         'preDFA', 'hearingsDummies', 'settingDummies', ''
         'Old Record','Others'], axis = 1)

DFA_final.dtypes


# In[64]:


y_DFA = DFA_final['DFA'].astype('category').cat.codes

DFA_final['intercept'] = 1.0

X_DFA = DFA_final.drop(columns = ['DFA','FailureToAppear','DFAAfter','DFABefore',
                                 'AgeGroup_<19','eDiscovery','AdultMisdemeanor',
                                 'AgeGroup_30-39','JuvenileFelony','JuvenileMisdemeanor',
                                 'Gender_Unknown','Venue_Dummies_Other Venues',
                                 'Venue_Dummies_Seattle Venue','Police_Dummies_Agent7'], axis=1)
X_DFA3 = DFA_final.drop(columns = ['DFA','FailureToAppear','DFAAfter','DFABefore'], axis =1)

#THis is variable from Decision Tree:
X_DFA3 = X_DFA3[['Seriousness','VUCSA','AdultFelony','Domestic violence','Highest Class_M',
                    'Car Theft Initiative', 'settingDummies_1','Gun case',
                    'Violent Level_Violent Charge','Previous DFA_1']]
# Validation: Holdout method
X_trainDFA3, X_testDFA3, y_trainDFA3, y_testDFA3 = train_test_split(X_DFA3, y_DFA,
                                                test_size=0.20,random_state=109) 


# In[65]:


scaler = StandardScaler()  
scaler.fit(X_trainDFA3)
X_trainDFA3 = scaler.transform(X_trainDFA3)  
X_testDFA3 = scaler.transform(X_testDFA3)

accuracy1 = []
precision1 = []
recall1 = []
f11 = []
xsize = X_trainDFA3.shape
step = 1
xaxis = range(1,10,step)

actFunc = ['identity', 'logistic', 'tanh', 'relu']
for x in xaxis:
    ## Multi-layer perceptron classification - one hidden layer of 3 neurons
    mlp = MLPClassifier(hidden_layer_sizes=(x), max_iter=1000,random_state = 109, activation = 'identity')  
    mlp.fit(X_trainDFA3, y_trainDFA3)  

    ## predict test set 
    y_predDFA3 = mlp.predict(X_testDFA3)

    accuracy1.append(metrics.accuracy_score(y_testDFA3,y_predDFA3)*100)
    precision1.append(metrics.precision_score(y_testDFA3,y_predDFA3)*100)
    recall1.append(metrics.recall_score(y_testDFA3,y_predDFA3)*100)
    f11.append(metrics.f1_score(y_testDFA3,y_predDFA3)*100)

maxAccuracyIndex = accuracy1.index(max(accuracy1))
maxPrecisionIndex = precision1.index(max(precision1))
maxRecallIndex = recall1.index(max(recall1))
maxF1Index = f11.index(max(f11))

print("max min_sample_leaf accuracy: " + str(round(max(accuracy1),2)) + " index " + str(maxAccuracyIndex*step+1))
print("max min_sample_leaf precision: " + str(round(max(precision1),2)) + " index " + str(maxPrecisionIndex*step+1))
print("max min_sample_leaf recall: " + str(round(max(recall1),2)) + " index " + str(maxRecallIndex*step+1))
print("max min_sample_leaf f1: " + str(round(max(f11),2)) + " index " + str(maxF1Index*step+1))


print("At max accuracy index " + str(maxAccuracyIndex*step+1) + 
      "\n Accuracy:  " + str(round(accuracy1[maxAccuracyIndex],2)) + "%"
      "\n Precision:  " + str(round(precision1[maxAccuracyIndex],2)) + "%"
      "\n Recall:  " + str(round(recall1[maxAccuracyIndex],2))+ "%" +
      "\n F1:  " + str(round(f11[maxAccuracyIndex],2))+ "%")

print("At max Precision index " + str(maxPrecisionIndex*step+1) + 
      "\n Accuracy:  " + str(round(accuracy1[maxPrecisionIndex],2)) + "%"
      "\n Precision:  " + str(round(precision1[maxPrecisionIndex],2)) + "%"
      "\n Recall:  " + str(round(recall1[maxPrecisionIndex],2))+ "%"+
      "\n F1:  " + str(round(f11[maxPrecisionIndex],2))+ "%") 

print("At max Recall index " + str(maxRecallIndex*step+1) + 
      "\n Accuracy:  " + str(round(accuracy1[maxRecallIndex],2)) + "%"
      "\n Precision:  " + str(round(precision1[maxRecallIndex],2)) + "%"
      "\n Recall:  " + str(round(recall1[maxRecallIndex],2))+ "%"+
      "\n F1:  " + str(round(f11[maxRecallIndex],2))+ "%")    

print("At max f1 index " + str(maxF1Index*step+1) + 
      "\n Accuracy:  " + str(round(accuracy1[maxF1Index],2)) + "%"
      "\n Precision:  " + str(round(precision1[maxF1Index],2)) + "%"
      "\n Recall:  " + str(round(recall1[maxF1Index],2))+ "%"+
      "\n F1:  " + str(round(f11[maxF1Index],2))+ "%")    
    


# #### Logistic Activation

# In[66]:


accuracy2 = []
precision2 = []
recall2 = []
f12 = []
xsize = X_trainDFA3.shape
step = 1
xaxis = range(1,10,step)

actFunc = ['identity', 'logistic', 'tanh', 'relu']
for x in xaxis:
    ## Multi-layer perceptron classification - one hidden layer of 3 neurons
    mlp = MLPClassifier(hidden_layer_sizes=(x), max_iter=1000,random_state = 109, activation = 'logistic')  
    mlp.fit(X_trainDFA3, y_trainDFA3)  

    ## predict test set 
    y_predDFA3 = mlp.predict(X_testDFA3)

    accuracy2.append(metrics.accuracy_score(y_testDFA3,y_predDFA3)*100)
    precision2.append(metrics.precision_score(y_testDFA3,y_predDFA3)*100)
    recall2.append(metrics.recall_score(y_testDFA3,y_predDFA3)*100)
    f12.append(metrics.f1_score(y_testDFA3,y_predDFA3)*100)

maxAccuracyIndex = accuracy2.index(max(accuracy2))
maxPrecisionIndex = precision2.index(max(precision2))
maxRecallIndex = recall2.index(max(recall2))
maxF1Index = f12.index(max(f12))

print("max hidden_layers accuracy: " + str(round(max(accuracy2),2)) + " index " + str(maxAccuracyIndex*step+1))
print("max hidden_layers precision: " + str(round(max(precision2),2)) + " index " + str(maxPrecisionIndex*step+1))
print("max hidden_layers recall: " + str(round(max(recall2),2)) + " index " + str(maxRecallIndex*step+1))
print("max hidden_layers f1: " + str(round(max(f12),2)) + " index " + str(maxF1Index*step+1))


print("At max accuracy index " + str(maxAccuracyIndex*step+1) + 
      "\n Accuracy:  " + str(round(accuracy2[maxAccuracyIndex],2)) + "%"
      "\n Precision:  " + str(round(precision2[maxAccuracyIndex],2)) + "%"
      "\n Recall:  " + str(round(recall2[maxAccuracyIndex],2))+ "%" +
      "\n F1:  " + str(round(f12[maxAccuracyIndex],2))+ "%")

print("At max Precision index " + str(maxPrecisionIndex*step+1) + 
      "\n Accuracy:  " + str(round(accuracy2[maxPrecisionIndex],2)) + "%"
      "\n Precision:  " + str(round(precision2[maxPrecisionIndex],2)) + "%"
      "\n Recall:  " + str(round(recall2[maxPrecisionIndex],2))+ "%"+
      "\n F1:  " + str(round(f12[maxPrecisionIndex],2))+ "%") 

print("At max Recall index " + str(maxRecallIndex*step+1) + 
      "\n Accuracy:  " + str(round(accuracy2[maxRecallIndex],2)) + "%"
      "\n Precision:  " + str(round(precision2[maxRecallIndex],2)) + "%"
      "\n Recall:  " + str(round(recall2[maxRecallIndex],2))+ "%"+
      "\n F1:  " + str(round(f12[maxRecallIndex],2))+ "%")    

print("At max f1 index " + str(maxF1Index*step+1) + 
      "\n Accuracy:  " + str(round(accuracy2[maxF1Index],2)) + "%"
      "\n Precision:  " + str(round(precision2[maxF1Index],2)) + "%"
      "\n Recall:  " + str(round(recall2[maxF1Index],2))+ "%"+
      "\n F1:  " + str(round(f12[maxF1Index],2))+ "%")   


# #### tanh Activation

# In[67]:


accuracy3 = []
precision3 = []
recall3 = []
f13 = []
xsize = X_trainDFA3.shape
step = 1
xaxis = range(1,10,step)

actFunc = ['identity', 'logistic', 'tanh', 'relu']
for x in xaxis:
    ## Multi-layer perceptron classification - one hidden layer of 3 neurons
    mlp = MLPClassifier(hidden_layer_sizes=(x), max_iter=1000,random_state = 109, activation = 'tanh')  
    mlp.fit(X_trainDFA3, y_trainDFA3)  

    ## predict test set 
    y_predDFA3 = mlp.predict(X_testDFA3)

    accuracy3.append(metrics.accuracy_score(y_testDFA3,y_predDFA3)*100)
    precision3.append(metrics.precision_score(y_testDFA3,y_predDFA3)*100)
    recall3.append(metrics.recall_score(y_testDFA3,y_predDFA3)*100)
    f13.append(metrics.f1_score(y_testDFA3,y_predDFA3)*100)

maxAccuracyIndex = accuracy3.index(max(accuracy3))
maxPrecisionIndex = precision3.index(max(precision3))
maxRecallIndex = recall3.index(max(recall3))
maxF1Index = f13.index(max(f13))

print("max hidden_layers accuracy: " + str(round(max(accuracy3),2)) + " index " + str(maxAccuracyIndex*step+1))
print("max hidden_layers precision: " + str(round(max(precision3),2)) + " index " + str(maxPrecisionIndex*step+1))
print("max hidden_layers recall: " + str(round(max(recall3),2)) + " index " + str(maxRecallIndex*step+1))
print("max hidden_layers f1: " + str(round(max(f13),2)) + " index " + str(maxF1Index*step+1))


print("At max accuracy index " + str(maxAccuracyIndex*step+1) + 
      "\n Accuracy:  " + str(round(accuracy3[maxAccuracyIndex],2)) + "%"
      "\n Precision:  " + str(round(precision3[maxAccuracyIndex],2)) + "%"
      "\n Recall:  " + str(round(recall3[maxAccuracyIndex],2))+ "%" +
      "\n F1:  " + str(round(f13[maxAccuracyIndex],2))+ "%")

print("At max Precision index " + str(maxPrecisionIndex*step+1) + 
      "\n Accuracy:  " + str(round(accuracy3[maxPrecisionIndex],2)) + "%"
      "\n Precision:  " + str(round(precision3[maxPrecisionIndex],2)) + "%"
      "\n Recall:  " + str(round(recall3[maxPrecisionIndex],2))+ "%"+
      "\n F1:  " + str(round(f13[maxPrecisionIndex],2))+ "%") 

print("At max Recall index " + str(maxRecallIndex*step+1) + 
      "\n Accuracy:  " + str(round(accuracy3[maxRecallIndex],2)) + "%"
      "\n Precision:  " + str(round(precision3[maxRecallIndex],2)) + "%"
      "\n Recall:  " + str(round(recall3[maxRecallIndex],2))+ "%"+
      "\n F1:  " + str(round(f13[maxRecallIndex],2))+ "%")    

print("At max f1 index " + str(maxF1Index*step+1) + 
      "\n Accuracy:  " + str(round(accuracy3[maxF1Index],2)) + "%"
      "\n Precision:  " + str(round(precision3[maxF1Index],2)) + "%"
      "\n Recall:  " + str(round(recall3[maxF1Index],2))+ "%"+
      "\n F1:  " + str(round(f13[maxF1Index],2))+ "%")   
    


# #### relu Activation

# In[68]:


accuracy4 = []
precision4 = []
recall4 = []
f14 = []
xsize = X_trainDFA3.shape
step = 1
xaxis = range(1,10,step)

actFunc = ['identity', 'logistic', 'tanh', 'relu']
for x in xaxis:
    ## Multi-layer perceptron classification - one hidden layer of 3 neurons
    mlp = MLPClassifier(hidden_layer_sizes=(x), max_iter=1000,random_state = 109, activation = 'relu')  
    mlp.fit(X_trainDFA3, y_trainDFA3)  

    ## predict test set 
    y_predDFA3 = mlp.predict(X_testDFA3)

    accuracy4.append(metrics.accuracy_score(y_testDFA3,y_predDFA3)*100)
    precision4.append(metrics.precision_score(y_testDFA3,y_predDFA3)*100)
    recall4.append(metrics.recall_score(y_testDFA3,y_predDFA3)*100)
    f14.append(metrics.f1_score(y_testDFA3,y_predDFA3)*100)

maxAccuracyIndex = accuracy4.index(max(accuracy4))
maxPrecisionIndex = precision4.index(max(precision4))
maxRecallIndex = recall4.index(max(recall4))
maxF1Index = f14.index(max(f14))

print("max hidden_layers accuracy: " + str(round(max(accuracy4),2)) + " index " + str(maxAccuracyIndex*step+1))
print("max hidden_layers precision: " + str(round(max(precision4),2)) + " index " + str(maxPrecisionIndex*step+1))
print("max hidden_layers recall: " + str(round(max(recall4),2)) + " index " + str(maxRecallIndex*step+1))
print("max hidden_layers f1: " + str(round(max(f14),2)) + " index " + str(maxF1Index*step+1))


print("At max accuracy index " + str(maxAccuracyIndex*step+1) + 
      "\n Accuracy:  " + str(round(accuracy4[maxAccuracyIndex],2)) + "%"
      "\n Precision:  " + str(round(precision4[maxAccuracyIndex],2)) + "%"
      "\n Recall:  " + str(round(recall4[maxAccuracyIndex],2))+ "%" +
      "\n F-stats:  " + str(round(f14[maxAccuracyIndex],2))+ "%")

print("At max Precision index " + str(maxPrecisionIndex*step+1) + 
      "\n Accuracy:  " + str(round(accuracy4[maxPrecisionIndex],2)) + "%"
      "\n Precision:  " + str(round(precision4[maxPrecisionIndex],2)) + "%"
      "\n Recall:  " + str(round(recall4[maxPrecisionIndex],2))+ "%"+
      "\n F-stats:  " + str(round(f14[maxPrecisionIndex],2))+ "%") 

print("At max Recall index " + str(maxRecallIndex*step+1) + 
      "\n Accuracy:  " + str(round(accuracy4[maxRecallIndex],2)) + "%"
      "\n Precision:  " + str(round(precision4[maxRecallIndex],2)) + "%"
      "\n Recall:  " + str(round(recall4[maxRecallIndex],2))+ "%"+
      "\n F-stats:  " + str(round(f14[maxRecallIndex],2))+ "%")    

print("At max f1 index " + str(maxF1Index*step+1) + 
      "\n Accuracy:  " + str(round(accuracy4[maxF1Index],2)) + "%"
      "\n Precision:  " + str(round(precision4[maxF1Index],2)) + "%"
      "\n Recall:  " + str(round(recall4[maxF1Index],2))+ "%"+
      "\n F-stats:  " + str(round(f14[maxF1Index],2))+ "%")   


# In[75]:


fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2,figsize=(20,20))
ax1.plot( xaxis, accuracy1,   color='skyblue', linewidth=2, label = 'accuracy')
ax1.plot( xaxis, precision1,   color='green', linewidth=2, label = 'precision')
ax1.plot( xaxis, recall1,   color='red', linewidth=2,  label="recall")
ax1.plot( xaxis, f11,   color='purple', linewidth=2,  label="recall")
ax1.set_title('Identity Activation Function')
ax1.set_xlabel('Hidden Layers')
ax1.set_ylabel('Percentage')
ax1.set_yticks(np.arange(0, 100, 5.0))
ax1.set_xticks(np.arange(min(xaxis)-1, max(xaxis)+1, 1))


ax2.plot( xaxis, accuracy2,   color='skyblue', linewidth=2, label = 'accuracy')
ax2.plot( xaxis, precision2,   color='green', linewidth=2, label = 'precision')
ax2.plot( xaxis, recall2,   color='red', linewidth=2,  label="recall")
ax2.plot( xaxis, f12,   color='purple', linewidth=2,  label="recall")
ax2.set_title('Logistic Activation Function')
ax2.set_xlabel('Hidden Layers')
ax2.set_ylabel('Percentage')
ax2.set_yticks(np.arange(0, 100, 5.0))
ax2.set_xticks(np.arange(min(xaxis)-1, max(xaxis)+1, 1))

ax3.plot( xaxis, accuracy3,   color='skyblue', linewidth=2, label = 'accuracy')
ax3.plot( xaxis, precision3,   color='green', linewidth=2, label = 'precision')
ax3.plot( xaxis, recall3,   color='red', linewidth=2,  label="recall")
ax3.plot( xaxis, f13,   color='purple', linewidth=2,  label="recall")
ax3.set_title('TanH Activation Function')
ax3.set_xlabel('Hidden Layers')
ax3.set_ylabel('Percentage')
ax3.set_yticks(np.arange(0, 100, 5.0))
ax3.set_xticks(np.arange(min(xaxis)-1, max(xaxis)+1, 1))

ax4.plot( xaxis, accuracy4,   color='skyblue', linewidth=2, label = 'accuracy')
ax4.plot( xaxis, precision4,   color='green', linewidth=2, label = 'precision')
ax4.plot( xaxis, recall4,   color='red', linewidth=2,  label="recall")
ax4.plot( xaxis, f14,   color='purple', linewidth=2,  label="recall")
ax4.set_title('ReLu Activation Function')
ax4.set_xlabel('Hidden Layers')
ax4.set_ylabel('Percentage')
ax4.set_yticks(np.arange(0, 100, 5.0))
ax4.set_xticks(np.arange(min(xaxis)-1, max(xaxis)+1, 1))

plt.suptitle("NN using variables from Decision Tree", fontsize=30)


# In[ ]:





# In[ ]:




