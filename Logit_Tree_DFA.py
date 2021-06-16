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


# In[2]:


DFA_train = pd.read_excel(r'/Users/evanko/Documents/Seattle University/BUAN 5510/Logistic Regression/DFAtrain_update.xlsx')

DFA_test = pd.read_excel(r'/Users/evanko/Documents/Seattle University/BUAN 5510/Logistic Regression/DFA_Test_2.xlsx')

DFA_test['Violent Level'] = np.where(DFA_test['Violent'] == 0, 'Non-violent Charge', 'Violent Charge')

DFA_final = pd.concat([DFA_train, DFA_test], ignore_index=True, sort=False)
DFA_final.dropna(subset=['Seriousness'], inplace=True)

#DFA_final = DFA_final.drop(['Unnamed: 0'], axis = 1)


# In[3]:


# Create dummies for train data
cat_vars=['AgeGroup','Gender','Venue_Dummies', 'Police_Dummies','Highest Class',
          'Violent Level','Previous DFA', 'hearingsDummies', 'settingDummies', 'Old Record']
for var in cat_vars:
    cat_list='var'+'_'+var
    cat_list = pd.get_dummies(DFA_final[var], drop_first = True, prefix=var)
    data1=DFA_final.join(cat_list)
    DFA_final=data1


# In[4]:


DFA_final = DFA_final.drop(columns = ['File Number','Defendant ID','DOB Anon','Referral Date','Case Types',
         'Event Code','Event Enter Date','Police Agency','Anon LE Number',
         'Venue','Age','Case No 1','Case No 2','Case No 3','Case No 4',
         'Case No 5','Case No 6','Violent','Previous DFA','Gender', 'AgeGroup',
         'Venue_Dummies','Police_Dummies', 'Highest Class', 'Violent Level',
         'preDFA', 'hearingsDummies', 'settingDummies', ''
         'Old Record'], axis = 1)

DFA_final.dtypes


# In[5]:


# In[69]:


sns.set(style="white")
corr = DFA_final.corr()
# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# In[6]:


y_DFA = DFA_final['DFA'].astype('category').cat.codes

DFA_final['intercept'] = 1.0

X_DFA = DFA_final.drop(columns = ['DFA','FailureToAppear','DFAAfter','DFABefore',
                                 'AgeGroup_<19','Others','eDiscovery','AdultMisdemeanor',
                                 'AgeGroup_30-39','JuvenileFelony','JuvenileMisdemeanor',
                                 'Gender_Unknown','Venue_Dummies_Other Venues',
                                 'Venue_Dummies_Seattle Venue','Police_Dummies_Agent7'], axis=1)


k = X_DFA

logit_DFA = sm.Logit(y_DFA, k).fit()
logit_DFA.summary()

# Validation: Holdout method
X_trainDFA, X_testDFA, y_trainDFA, y_testDFA = train_test_split(X_DFA, y_DFA,
                                                test_size=0.20,random_state=109) 
#Model 1:
# Choose Threshold = 0.5
logit_DFA1 = sm.Logit(y_trainDFA, X_trainDFA).fit()
print(logit_DFA1.summary())
y_predDFA1 = logit_DFA1.predict(X_testDFA)
y_predDFA1 = [ 0 if x < 0.5 else 1 for x in y_predDFA1]



#Confusion matrix
print(metrics.confusion_matrix(y_testDFA, y_predDFA1))

print ("Accuracy: " + str( metrics.accuracy_score(y_testDFA,y_predDFA1)*100)+" %")
print ("Precision: "+ str( metrics.precision_score(y_testDFA,y_predDFA1)*100) + " %")
print ( "Recall: "+ str( metrics.recall_score(y_testDFA,y_predDFA1)*100) +" %")

### Decision Tree model

X_DFA2 = DFA_final.drop(columns = ['DFA','FailureToAppear','DFAAfter','DFABefore'], axis =1)

# Validation: Holdout method
X_trainDFA2, X_testDFA2, y_trainDFA2, y_testDFA2 = train_test_split(X_DFA2, y_DFA,
                                                test_size=0.20,random_state=109) 
# Decison tree model 
clf = tree.DecisionTreeClassifier(class_weight=None, criterion='gini', 
                                  max_depth=10, max_features=None, 
                                  max_leaf_nodes=None, min_samples_leaf=10, 
                                  min_samples_split=2, min_weight_fraction_leaf=0.0, 
                                  presort=False, random_state=100, splitter='best')
clf = clf.fit(X_trainDFA2, y_trainDFA2)

y_predDFA2 = clf.predict(X_testDFA2)  
    
    # Confusion Matrix - Decision Tree Model 
print(metrics.confusion_matrix(y_testDFA2, y_predDFA2))

print ("Accuracy: " + str( metrics.accuracy_score(y_testDFA2,y_predDFA2)*100)+" %")
print ("Precision: "+ str( metrics.precision_score(y_testDFA2,y_predDFA2)*100) + " %")
print ( "Recall: "+ str( metrics.recall_score(y_testDFA2,y_predDFA2)*100) +" %")


# In[32]:


### Decision Tree model

X_DFA3 = DFA_final.drop(columns = ['DFA','FailureToAppear','DFAAfter','DFABefore'], axis =1)
# Validation: Holdout method
X_trainDFA3, X_testDFA3, y_trainDFA3, y_testDFA3 = train_test_split(X_DFA3, y_DFA,
                                                test_size=0.20,random_state=109) 
accuracy = []
precision = []
recall = []
xaxis = range(1,25,2)
for x in range(1,25,2):

    # Decison tree model 
    clf = tree.DecisionTreeClassifier(class_weight=None, criterion='gini', 
                                      max_depth=x, max_features=None, 
                                      max_leaf_nodes=None, min_samples_leaf=10, 
                                      min_samples_split=2, min_weight_fraction_leaf=0.0, 
                                      random_state=100, splitter='best')
    clf = clf.fit(X_trainDFA3, y_trainDFA3)

    y_predDFA3 = clf.predict(X_testDFA3)  
    
    accuracy.append(metrics.accuracy_score(y_testDFA3,y_predDFA3)*100)
    precision.append(metrics.precision_score(y_testDFA3,y_predDFA3)*100)
    recall.append(metrics.recall_score(y_testDFA3,y_predDFA3)*100)
        # Confusion Matrix - Decision Tree Model 
#     print(metrics.confusion_matrix(y_testDFA3, y_predDFA3))

#     print ("Max Depth: " + str(x) + " " +"Accuracy: " + str( metrics.accuracy_score(y_testDFA3,y_predDFA3)*100)+" %")
#     print ("Max Depth: " + str(x) + " " +"Precision: "+ str( metrics.precision_score(y_testDFA3,y_predDFA3)*100) + " %")
#     print ( "Max Depth: " + str(x) +" " + "Recall: "+ str( metrics.recall_score(y_testDFA3,y_predDFA3)*100) +" %")
plt.plot( xaxis, accuracy,   color='skyblue', linewidth=2, label = 'accuracy')
plt.plot( xaxis, precision,   color='green', linewidth=2, label = 'precision')
plt.plot( xaxis, recall,   color='red', linewidth=2,  label="recall")
plt.title('Effect of Max Depth')
plt.xlabel('Max Depth')
plt.ylabel('Percentage')
plt.yticks(np.arange(0, 100, 5.0))
plt.xticks(np.arange(min(xaxis), max(xaxis)+1, 1.0))
plt.legend()


# ### Based on above code 9 is best max depth

# In[33]:


### Decision Tree model

X_DFA3 = DFA_final.drop(columns = ['DFA','FailureToAppear','DFAAfter','DFABefore'], axis =1)
# Validation: Holdout method
X_trainDFA3, X_testDFA3, y_trainDFA3, y_testDFA3 = train_test_split(X_DFA3, y_DFA,
                                                test_size=0.20,random_state=109) 
accuracy = []
precision = []
recall = []
xaxis = range(1,25,2)
for x in range(1,25,2):

    # Decison tree model 
    clf = tree.DecisionTreeClassifier(class_weight=None, criterion='gini', 
                                      max_depth=9 #choosen by above
                                      , max_features=None, 
                                      max_leaf_nodes=None, min_samples_leaf=x, 
                                      min_samples_split=2, min_weight_fraction_leaf=0.0, 
                                      random_state=100, splitter='best')
    clf = clf.fit(X_trainDFA3, y_trainDFA3)

    y_predDFA3 = clf.predict(X_testDFA3)  
    
    accuracy.append(metrics.accuracy_score(y_testDFA3,y_predDFA3)*100)
    precision.append(metrics.precision_score(y_testDFA3,y_predDFA3)*100)
    recall.append(metrics.recall_score(y_testDFA3,y_predDFA3)*100)
        # Confusion Matrix - Decision Tree Model 
#     print(metrics.confusion_matrix(y_testDFA3, y_predDFA3))

#     print ("Max Depth: " + str(x) + " " +"Accuracy: " + str( metrics.accuracy_score(y_testDFA3,y_predDFA3)*100)+" %")
#     print ("Max Depth: " + str(x) + " " +"Precision: "+ str( metrics.precision_score(y_testDFA3,y_predDFA3)*100) + " %")
#     print ( "Max Depth: " + str(x) +" " + "Recall: "+ str( metrics.recall_score(y_testDFA3,y_predDFA3)*100) +" %")
plt.plot( xaxis, accuracy,   color='skyblue', linewidth=2, label = 'accuracy')
plt.plot( xaxis, precision,   color='green', linewidth=2, label = 'precision')
plt.plot( xaxis, recall,   color='red', linewidth=2,  label="recall")
plt.title('Effect of min_samples_leaf')
plt.xlabel('min_samples_leaf')
plt.ylabel('Percentage')
plt.yticks(np.arange(0, 100, 5.0))
plt.xticks(np.arange(min(xaxis), max(xaxis)+1, 1.0))
plt.legend()


# In[35]:


### Decision Tree model

X_DFA3 = DFA_final.drop(columns = ['DFA','FailureToAppear','DFAAfter','DFABefore'], axis =1)
# Validation: Holdout method
X_trainDFA3, X_testDFA3, y_trainDFA3, y_testDFA3 = train_test_split(X_DFA3, y_DFA,
                                                test_size=0.20,random_state=109) 
accuracy = []
precision = []
recall = []
xaxis = range(1,25,2)
for x in range(2,25,2):

    # Decison tree model 
    clf = tree.DecisionTreeClassifier(class_weight=None, criterion='gini', 
                                      max_depth=9 #choosen by above
                                      , max_features=None, 
                                      max_leaf_nodes=None, min_samples_leaf=1, 
                                      min_samples_split=x, min_weight_fraction_leaf=0.0, 
                                      random_state=100, splitter='best')
    clf = clf.fit(X_trainDFA3, y_trainDFA3)

    y_predDFA3 = clf.predict(X_testDFA3)  
    
    accuracy.append(metrics.accuracy_score(y_testDFA3,y_predDFA3)*100)
    precision.append(metrics.precision_score(y_testDFA3,y_predDFA3)*100)
    recall.append(metrics.recall_score(y_testDFA3,y_predDFA3)*100)
        # Confusion Matrix - Decision Tree Model 
#     print(metrics.confusion_matrix(y_testDFA3, y_predDFA3))

#     print ("Max Depth: " + str(x) + " " +"Accuracy: " + str( metrics.accuracy_score(y_testDFA3,y_predDFA3)*100)+" %")
#     print ("Max Depth: " + str(x) + " " +"Precision: "+ str( metrics.precision_score(y_testDFA3,y_predDFA3)*100) + " %")
#     print ( "Max Depth: " + str(x) +" " + "Recall: "+ str( metrics.recall_score(y_testDFA3,y_predDFA3)*100) +" %")
plt.plot( xaxis, accuracy,   color='skyblue', linewidth=2, label = 'accuracy')
plt.plot( xaxis, precision,   color='green', linewidth=2, label = 'precision')
plt.plot( xaxis, recall,   color='red', linewidth=2,  label="recall")
plt.title('Effect of min_samples_split')
plt.xlabel('min_samples_split')
plt.ylabel('Percentage')
plt.yticks(np.arange(0, 100, 5.0))
plt.xticks(np.arange(min(xaxis), max(xaxis)+1, 1.0))
plt.legend()


# In[39]:


# Decison tree model 
clf = tree.DecisionTreeClassifier(class_weight=None, criterion='gini', 
                                  max_depth=9 #choosen by above
                                  , max_features=None, 
                                  max_leaf_nodes=None, min_samples_leaf=1, 
                                  min_samples_split=2, min_weight_fraction_leaf=0.0, 
                                  random_state=100, splitter='best')
clf = clf.fit(X_trainDFA3, y_trainDFA3)

y_predDFA3 = clf.predict(X_testDFA3)  

#     accuracy.append(metrics.accuracy_score(y_testDFA3,y_predDFA3)*100)
#     precision.append(metrics.precision_score(y_testDFA3,y_predDFA3)*100)
#     recall.append(metrics.recall_score(y_testDFA3,y_predDFA3)*100)
#Confusion Matrix - Decision Tree Model 
print(metrics.confusion_matrix(y_testDFA3, y_predDFA3))

print ("Accuracy: " + str( round(metrics.accuracy_score(y_testDFA3,y_predDFA3)*100,2))+" %")
print ("Precision: "+ str( round(metrics.precision_score(y_testDFA3,y_predDFA3)*100,2)) + " %")
print ("Recall: "+ str( round(metrics.recall_score(y_testDFA3,y_predDFA3)*100,2)) +" %")


# In[44]:


# Decison tree model 
clf = tree.DecisionTreeClassifier(class_weight=None, criterion='gini', 
                                  max_depth=13 #choosen by above
                                  , max_features=None, 
                                  max_leaf_nodes=None, min_samples_leaf=1, 
                                  min_samples_split=2, min_weight_fraction_leaf=0.0, 
                                  random_state=100, splitter='best')
clf = clf.fit(X_trainDFA3, y_trainDFA3)

y_predDFA3 = clf.predict(X_testDFA3)  

#     accuracy.append(metrics.accuracy_score(y_testDFA3,y_predDFA3)*100)
#     precision.append(metrics.precision_score(y_testDFA3,y_predDFA3)*100)
#     recall.append(metrics.recall_score(y_testDFA3,y_predDFA3)*100)
#Confusion Matrix - Decision Tree Model 
print(metrics.confusion_matrix(y_testDFA3, y_predDFA3))

print ("Accuracy: " + str( round(metrics.accuracy_score(y_testDFA3,y_predDFA3)*100,2))+" %")
print ("Precision: "+ str( round(metrics.precision_score(y_testDFA3,y_predDFA3)*100,2)) + " %")
print ("Recall: "+ str( round(metrics.recall_score(y_testDFA3,y_predDFA3)*100,2)) +" %")


# In[43]:


tree.export_graphviz(clf, out_file=r'/Users/evanko/Documents/Seattle University/BUAN 5510/Logistic Regression/Dtree.dot', feature_names=X_trainDFA2.columns, class_names = True)


# In[41]:


text_representation = tree.export_text(clf)
print(text_representation)


# In[ ]:




