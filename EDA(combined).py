#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import date 
import warnings

from sklearn.model_selection import train_test_split
import missingno as miss
import time

warnings.filterwarnings("ignore")


# ### Import All Data and split the data for training and test purposes (80%/20%)

#create a map of all data files 
data1 = pd.read_excel('Data for SU Part 1 - Filed Cases and Hearings.xlsx', sheet_name = None)
data2 = pd.read_excel('Data for SU Part 2.xlsx', sheet_name = None)
data4 = pd.read_excel('Data for SU Part 4 - cleaned criminal history and updated hearing key.xlsx', sheet_name = None)
data5 = pd.read_excel('Data for SU Part 5 - Custody history with time stamps.xlsx', sheet_name = None)


# In[8]:


#get data tables for first file
FiledCasesSU = pd.read_excel('FiledCasesSUTest.xlsx')
caseEventsSU = pd.read_excel('caseEventsSUTest_Fill.xlsx')
CaseEventKey1 = data1['CaseEvent Key']
CaseTypeKey = data1['CaseType Key']

#get data tables for 2nd file
SUCharges = data2['SU Charges']
ChargeDefinitions = data2['Charge Definitions']
DispositionKey = data2['Disposition Key']
SUCrimHist2 = data2['SU CrimHist']

#get data for 4th file
SUCrimHist4 = data4['SU CrimHist']
CaseEventKey4 = data4['CaseEvent Key']

#get the data for 5th file
CustodyStatusTimeStamp = data5['CustodyStatusTimeStamp']


# In[9]:


CustodyStatusTimeStamp = CustodyStatusTimeStamp.drop_duplicates(subset=['Defendant_ID','Status_History_Date'], keep = 'last')
uniqueDiD = caseEventsSU['Defendant ID'].unique()
#get the training and test defendant ID
dID_train, dID_test = train_test_split(uniqueDiD, test_size=0.2, random_state=41)
#find related FiledCases to the training and test defendant ID
fn_train = FiledCasesSU[FiledCasesSU['Defendant ID'].isin(dID_train)]['File Number']
fn_test = FiledCasesSU[FiledCasesSU['Defendant ID'].isin(dID_test)]['File Number']


# In[10]:


#get data tables for first file
FiledCasesSUTrain = FiledCasesSU[FiledCasesSU['Defendant ID'].isin(dID_train)]
caseEventsSUTrain = caseEventsSU[caseEventsSU['Defendant ID'].isin(dID_train)]

#get data tables for 2nd file
SUcharge = SUCharges[SUCharges['File Number'].isin(fn_train)]

SUCrimHist2Train = SUCrimHist2[SUCrimHist2['DefendantID'].isin(dID_train)]

#get data for 4th file
SUCrimHist4Train = SUCrimHist4[SUCrimHist4['DefendantID'].isin(dID_train)]

#get the data for 5th file
CustodyStatusTimeStampTrain = CustodyStatusTimeStamp[CustodyStatusTimeStamp['Defendant_ID'].isin(dID_train)]


# In[11]:


tot = FiledCasesSU[FiledCasesSU['Defendant ID'].isin(uniqueDiD)].shape[0]
tr =FiledCasesSU[FiledCasesSU['Defendant ID'].isin(dID_train)].shape[0]
te =FiledCasesSU[FiledCasesSU['Defendant ID'].isin(dID_test)].shape[0]
print(tr/tot,te/tot)



# In[12]:


fn_train


# ### EDA 0.1

# In[13]:


print(uniqueDiD.size,dID_train.size,dID_test.size)
print(caseEventsSU[caseEventsSU['Defendant ID'].isin(uniqueDiD)].shape,caseEventsSU[caseEventsSU['Defendant ID'].isin(dID_train)].shape,caseEventsSU[caseEventsSU['Defendant ID'].isin(dID_test)].shape)


# In[14]:


#grab a subset of the data that has a Event_Docket_Date but is missing the Defendant_Event_Status
fillableData = caseEventsSUTrain[pd.notnull(caseEventsSU.Event_Docket_Date) & pd.isnull(caseEventsSU.Defendant_Event_Status)]
k = fillableData['Defendant ID'].unique()
k.size


# In[15]:


#remove all of the outliers or misinputStatus_History_Date
print(CustodyStatusTimeStampTrain.shape)
CustodyStatusTimeStampTrain = CustodyStatusTimeStampTrain[CustodyStatusTimeStampTrain['Status_History_Date'] != '1/1/1753 12:00:00 AM']
print(CustodyStatusTimeStampTrain.shape)


# In[16]:


#function to use to get the status dependant on past and future status
def statusGen(past,future):
    if(past.upper() == 'IN'):
        return 'IN'
    else:
        return 'OUT'


# In[17]:


start = time.time()
#create dictionary to hold the results
statusDict = {}
#iterate through each defendant id
for dID in k:
    try:
        #get subset matching the defendant ID
        l1 = fillableData[fillableData['Defendant ID'] == dID]
        l2 = CustodyStatusTimeStampTrain[CustodyStatusTimeStampTrain['Defendant_ID'] == dID]

        #get the date columns we need
        t1 = pd.to_datetime(l1['Event_Docket_Date'])
        t2 = pd.to_datetime(l2['Status_History_Date'])


        #for each date in Event_Docket_Date get the new status
        for h in t1:
            #split t2 into the past and future sub datasets
            t2Past = t2[t2 < h]
            t2Future = t2[t2 > h]
            
            #if there are any data in the past find the closest date and lookup the status
            if(t2Past.size > 0):
                l3Past = min(t2Past, key=lambda d: abs(datetime.strptime(str(d), "%Y-%m-%d %H:%M:%S") - datetime.strptime(str(h), "%Y-%m-%d %H:%M:%S")))
                statusPast = l2[l2['Status_History_Date'] == l3Past]['Custody Status History'].item()
            else:
                #status is out if there was not any date before
                statusPast = 'OUT'
            #if there are any data in the future find the closest date and lookup the status
            if(t2Future.size > 0):
                l3Future = min(t2Future, key=lambda d: abs(datetime.strptime(str(d), "%Y-%m-%d %H:%M:%S") - datetime.strptime(str(h), "%Y-%m-%d %H:%M:%S")))
                statusFuture = l2[l2['Status_History_Date'] == l3Future]['Custody Status History'].item()
            else:
                #default to the past if there is nothing in front
                statusFuture = statusPast

            #fill dictionary Key =tuple of (Defendant ID, Event_Docket_Date) value = status
            statusDict[(dID,h)] = statusGen(statusPast,statusFuture)
    except Exception as e:
        print('ERROR at index {}: {}'.format(dID, h))
        print(e)
        print(caseEventsSUTrain[(caseEventsSUTrain['Defendant ID'] == dID) & (caseEventsSUTrain['Event_Docket_Date'] == h)].shape)
        
elapsed_time_fl = (time.time() - start)
print(elapsed_time_fl)


# In[18]:


#add tuple column for easier merge
caseEventsSUTrain['Tuple'] = list(zip(caseEventsSUTrain['Defendant ID'],caseEventsSUTrain['Event_Docket_Date']))
#convert status dictionary to dataframe
statusDF = pd.DataFrame.from_dict(statusDict,orient='index').reset_index()
#rename dataframe columns
statusDF.columns = ['Tuple','value1']
#check shape
statusDF.shape


# In[19]:


start = time.time()
#fill the missing values
#create dataframe and merge with caseEventsSUTrain and statusDF
caseEventsSUTrain = pd.merge(left = caseEventsSUTrain, right = statusDF, left_on = 'Tuple',right_on = 'Tuple',how = 'left')
#fill missing values from statusDF['value1']
caseEventsSUTrain['Defendant_Event_Status'] = caseEventsSUTrain['Defendant_Event_Status'].fillna(caseEventsSUTrain['value1'])
#fill the remaining missing values with OUT
caseEventsSUTrain['Defendant_Event_Status'] = caseEventsSUTrain['Defendant_Event_Status'].fillna('OUT')
#drop the not needed columns (Tuple and Value1)
caseEventsSUTrain = caseEventsSUTrain.drop(['Tuple','value1'], axis = 1)
elapsed_time_fl = (time.time() - start)
print(elapsed_time_fl)


# In[20]:


#check to see all missing values are filled
caseEventsSUTrain.isnull().sum()


# In[21]:


#export to datafile
caseEventsSUTrain.to_excel("caseEventsSUTrainFill.xlsx",index = False) 


# In[ ]:


FiledCasesSU.dtypes


# ### EDA 0.2

#Visualize missing values for FileCase: 
print(" \nCount total NaN at each column in FiledCasesSU  :\n",FiledCasesSU.isnull().sum())

miss.matrix(FiledCasesSU)

#Visualize missing values for CaseEvent:
print(" \nCount total NaN at each column in caseEventsSU  :\n",
      caseEventsSU.isnull().sum())

miss.matrix(caseEventsSU)

#Visualize missing values for CaseEvent Key:
print(" \nCount total NaN at each column in CaseEvent Key :\n",
      CaseEventKey1.isnull().sum())

miss.matrix(CaseEventKey1)

#Visualize missing values for CaseType Key:
print(" \nCount total NaN at each column in CaseType Key :\n",
      CaseTypeKey.isnull().sum())

miss.matrix(CaseTypeKey)

#Visualize missing values for Charges:
print(" \nCount total NaN at each column in Charges Key :\n",
      SUCharges.isnull().sum())

miss.matrix(SUCharges)

#Visualize missing values for ChargeDef:
print(" \nCount total NaN at each column in Charges Definition :\n",
      ChargeDefinitions.isnull().sum())

miss.matrix(ChargeDefinitions)

#Visualize missing values for Diposition Key:
print(" \nCount total NaN at each column in Diposition Key :\n",
      DispositionKey.isnull().sum())

miss.matrix(DispositionKey)

#Visualize missing values for Criminal History:
print(" \nCount total NaN at each column in Crim Hist :\n",
      SUCrimHist2.isnull().sum())

miss.matrix(SUCrimHist2) 

# Table 5: Data for SU Part 5 - Custody history with time stamps
print(" \nCount total NaN at each column in CustodyStatusTimeStamp :\n", CustodyStatusTimeStamp.isnull().sum()) 

miss.matrix(CustodyStatusTimeStamp) 


# In[ ]:
# ### EDA 0.3 

# Find the list of class U and I and combine them  
list_U = np.where(ChargeDefinitions['Class'] == 'U') 
list_I = np.where(ChargeDefinitions['Class'] == 'I')  

list_UandI = np.concatenate((list_U, list_I), axis=None)

# Get the codes that have indexes of class U and I
# Convert those codes to string and put them in a list
codes_UandI = list()
for UandI in list_UandI:
    codes_UandI.append(str(ChargeDefinitions['Code'][UandI]))

# Pad the 4-digit codes with 0, put them in a list 
# Codes with letters are left alone since they're 5 digits already 
codes_UandI_list = list()
for codes in codes_UandI:        
    codes_UandI_list.append(codes.rjust(5, "0"))
    
# Drop nan values in current charges column for now 
SUcharge = SUcharge.dropna(subset=['Current Charge(s)'])

# Split the text after "-" in current charges, only codes left
list_codes_notext = list()  # Put the codes in a seperate list
for cur_charges in SUcharge['Current Charge(s)']:
    codes_notext = cur_charges.split("-")  
    list_codes_notext.append(codes_notext[0].rstrip())
    
#print("codes_UandI_list: {}".format(codes_UandI_list))
#print("list_codes_notext: {}".format(list_codes_notext))

# Append the list of codes to charge table column  
SUcharge['Current_Charge_Codes'] = np.asarray(list_codes_notext)

# Run a loop to remove rows indexes that have current_codes == UandI_codes from the SUcharges table 
for UandI_codes in codes_UandI_list:
    SUcharge.drop(SUcharge[SUcharge['Current_Charge_Codes'] == UandI_codes].index, inplace = True) 


# In[ ]:


SUcharge.to_excel('SUcharge_clean.xlsx')


# In[ ]:


# ### Failure to appear

# In[8]:


caseEventsSUTrain.dtypes


# In[31]:
### Failure to appear

#dummy for DFA
caseEventsSUTrain['FailureToAppear'] = caseEventsSUTrain['Event_Code'].apply(lambda x: 1 if x == 'DEFFTA' else 0)


#Age def
def calculateAge(birthDate): 
    days_in_year = 365.2425    
    age = int((date.today() - birthDate).days / days_in_year) 
    return age 

#get Age
FiledCasesSUTrain['DOB Anon'] = pd.to_datetime(FiledCasesSUTrain['DOB Anon'])
FiledCasesSUTrain['Age'] = FiledCasesSUTrain['DOB Anon'].dt.date.apply(calculateAge)

#DFA
DFA = caseEventsSUTrain.groupby('File Number')['FailureToAppear'].sum()

FiledCasesSUTrain.shape

caseEventsSUTrain.shape

#join Filed Cases and DFA on File Number

DFA1 = pd.merge(left = FiledCasesSUTrain, right = DFA, left_on = 'File Number',right_on = 'File Number',how = 'left')
DFA1['DFA'] = np.where(DFA1['FailureToAppear'] > 0, 1, 0)    


failureRate = DFA1.groupby(['File Number'])['DFA'].sum()/DFA1.groupby(['File Number'])['DFA'].count()
failureRate


round(failureRate.mean()*100,2)

# In[ ]:
### 1.1 DFA Gender
failureRateGender = DFA1.groupby(['File Number','Gender'])['DFA'].sum()/DFA1.groupby(['File Number','Gender'])['DFA'].count()
failureRateGender = failureRateGender.apply(lambda x: round(x*100,2))

round(failureRateGender.mean()*100,2)

temp = failureRateGender.unstack('Gender').mean()
plt.bar(temp.index,temp)
plt.title('Failure to Appear by Gender')
plt.ylabel('Percentage')
plt.show()

# In[ ]:

### 1.2 DFA Age group
bins= [0,19,30,40,50,60,150]
labels = ['<19','20-29','30-39','40-49','50-59','60+']
DFA1['AgeGroup'] = pd.cut(DFA1['Age'], bins=bins, labels=labels, right=False)


failureRateAge = DFA1.groupby(['File Number','AgeGroup'])['DFA'].sum()/DFA1.groupby(['File Number','AgeGroup'])['DFA'].count()
failureRateAge = failureRateAge.apply(lambda x: round(x*100,2))

temp = failureRateAge.unstack('AgeGroup').mean()

plt.bar(temp.index,temp)
plt.title('Failure to Appear by Age Group')
plt.ylabel('Percentage')
plt.show()

# In[ ]:

### 1.3 DFA rate by Venues
DFA1['Venue'].value_counts()
DFA1['Venue_Dummies'] = ""

#Add in the dummies based on Venue:
DFA1['Venue_Dummies'] = np.where(DFA1['Venue'] == 'SEA', 'Seattle Venue', 
                                 np.where(DFA1['Venue'] == 'KNT', 'Kent Venue', 'Other Venues'))
DFA1['Venue_Dummies'].value_counts()

failureRateVE = DFA1.groupby(['File Number','Venue_Dummies'])['DFA'].sum()/DFA1.groupby(['File Number','Venue_Dummies'])['DFA'].count()
failureRateVE = failureRateVE.apply(lambda x: round(x*100,2))

temp = failureRateVE.unstack('Venue_Dummies').mean()
plt.bar(temp.index,temp)
plt.title('Failure to Appear by Venue')
plt.ylabel('Percentage')
plt.show()

# In[ ]:

### 1.4 DFA by Police Agency

DFA1['Police_Dummies'] = ""

#Add in the dummies based on Police Agency:
DFA1['Police_Dummies'] = np.where(DFA1['Police Agency'] == 'Seattle Police Department','Agent1',
                                     np.where(DFA1['Police Agency'] == "King County Sheriff's Office", 'Agent2',
                                     np.where(DFA1['Police Agency'] == 'Auburn Police Department', 'Agent3',
                                     np.where(DFA1['Police Agency'] == 'Kent Police Department', 'Agent4',
                                     np.where(DFA1['Police Agency'] == 'Federal Way Police Department', 'Agent5',
                                     np.where(DFA1['Police Agency'] == 'Renton Police Department', 'Agent6',
                                     np.where(DFA1['Police Agency'] == 'Bellevue Police Department', 'Agent7', 'Others')))))))
DFA1['Police_Dummies'].shape
DFA1['Police Agency'].shape

#DFA by Police Agency:

failureRatePA = DFA1.groupby(['File Number','Police_Dummies'])['DFA'].sum()/DFA1.groupby(['File Number','Police_Dummies'])['DFA'].count()
failureRatePA = failureRatePA.apply(lambda x: round(x*100,2))

temp = failureRatePA.unstack('Police_Dummies').mean()
plt.bar(temp.index,temp)
plt.title('Failure to Appear by Police Agency')
plt.ylabel('Percentage')
plt.show()

# In[ ]:

### 1.5 Criminal history and conviction level
#group all of the outliers with AM: 

SUCrimHist2['ConvictionLevel'].value_counts()
SUCrimHist2Train['ConvictionLevel'].value_counts()
SUCrimHist2Train['ConvictionLevel'].shape
SUCrimHist2Train['Convic_Dummies'] = np.where(SUCrimHist2Train['ConvictionLevel'] == 'AF', 'AF',
                                              np.where(SUCrimHist2Train['ConvictionLevel'] == 'JM', 'JM',
                                              np.where(SUCrimHist2Train['ConvictionLevel'] == 'JF','JF','AM')))
#Merge Crimhist and DFA1:
DFA2= SUCrimHist2Train.merge(right = DFA1.drop_duplicates(subset=['Defendant ID']), left_on = 'DefendantID', right_on = 'Defendant ID', how ='left')
DFA2['DFA'].value_counts()
DFA2.count()

#Graph the DFA by conviction level
failureRateCL = DFA2.groupby(['Defendant ID','Convic_Dummies'])['DFA'].sum()/DFA2.groupby(['Defendant ID','Convic_Dummies'])['DFA'].count()
failureRateCL = failureRateCL.apply(lambda x: round(x*100,2))

temp = failureRateCL.unstack('Convic_Dummies').mean()
plt.bar(temp.index,temp)
plt.title('Failure to Appear by Conviction Level')
plt.ylabel('Percentage')
plt.show()
DFA2['Convic_Dummies'].value_counts()
#Show mising values in DFA2:
miss.matrix(DFA2)

#Make the colum Historical offense in Year:
DFA2['OffenseDate'] = pd.to_datetime(DFA2['OffenseDate'],errors='coerce')
DFA2 = DFA2.dropna(subset = ['OffenseDate']) #Drop missing values of Offense date- only 4%
DFA2['Historical Offense'] = DFA2['OffenseDate'].dt.date.apply( calculateAge)

#Plot the chart with both Criminal record and Convcition Level

conAM1 = DFA1.where((DFA2['Convic_Dummies'] == 'AM') & (DFA2['Historical Offense'] <= 5)).dropna()
conAF1 = DFA1.where((DFA2['Convic_Dummies'] == 'AF') & (DFA2['Historical Offense'] <= 5)).dropna()
conJM1 = DFA1.where((DFA2['Convic_Dummies'] == 'JM') & (DFA2['Historical Offense'] <= 5)).dropna()
conJF1 = DFA1.where((DFA2['Convic_Dummies'] == 'JF') & (DFA2['Historical Offense'] <= 5)).dropna()

conAM2 = DFA1.where((DFA2['Convic_Dummies'] == 'AM') & (DFA2['Historical Offense'] > 5)).dropna()
conAF2 = DFA1.where((DFA2['Convic_Dummies'] == 'AF') & (DFA2['Historical Offense'] > 5)).dropna()
conJM2 = DFA1.where((DFA2['Convic_Dummies'] == 'JM') & (DFA2['Historical Offense'] < 5)).dropna()
conJF2 = DFA1.where((DFA2['Convic_Dummies'] == 'JF') & (DFA2['Historical Offense'] > 5)).dropna()

DFAcon1 = round((conAM1['DFA'].sum()/conAM1['DFA'].count())*100,1)
DFAcon2 = round((conAF1['DFA'].sum()/conAF1['DFA'].count())*100,1)
DFAcon3 = round((conJM1['DFA'].sum()/conJM1['DFA'].count())*100,1)
DFAcon4 = round((conJF1['DFA'].sum()/conJF1['DFA'].count())*100,1)

DFAcon5 = round((conAM2['DFA'].sum()/conAM2['DFA'].count())*100,1)
DFAcon6 = round((conAF2['DFA'].sum()/conAF2['DFA'].count())*100,1)
DFAcon7 = round((conJM2['DFA'].sum()/conJM2['DFA'].count())*100,1)
DFAcon8 = round((conJF2['DFA'].sum()/conJF2['DFA'].count())*100,1)

DFA_con1 = [DFAcon1, DFAcon2, DFAcon3, DFAcon4]
DFA_con2 = [DFAcon5, DFAcon6, DFAcon7, 0]

labels = ['AM', 'AF', 'JM', 'JF']
x = np.arange(len(labels))  # the label locations 
width = 0.30  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x, DFA_con1, width, label='Recent')
rects2 = ax.bar(x + width, DFA_con2, width, label='Old')


# Add some text for labels, title and custom x-axis tick labels
ax.set_ylabel('Percentage')
ax.set_title('Failure to Appear by Criminal Record and Conviction Level')
ax.set_xticks(x + 0.5*width)
ax.set_xticklabels(labels)
ax.legend(handletextpad = 0.1, borderaxespad = 0.1)

def autolabel(rects):
    #Attach a text label above each bar in *rects*, displaying its height 
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)


fig.tight_layout()

plt.show()

# In[ ]:

# ### 1.6 DFA by Charge Classes and Seriousness Levels: 

### Find representative charge for each File Number 

# Drop nan values in class column 
ChargeDefinitions = ChargeDefinitions.dropna(subset=['Class'])

# Pad 0 to codes
ChargeDefinitions['Code'] = ChargeDefinitions['Code'].astype(str).apply(lambda x: x.zfill(5))

# Replace nan values by 0
ChargeDefinitions['Violent'] = ChargeDefinitions['Violent'].replace(np.nan, 0)
ChargeDefinitions['Seriousness'] = ChargeDefinitions['Seriousness'].replace(np.nan, 0)

NewSUCharges = SUcharge.drop_duplicates().merge(ChargeDefinitions.drop_duplicates(), left_on="Current_Charge_Codes", right_on="Code")

# Add up seriousness and violent levels 
grouped_SUcharge = NewSUCharges.groupby(['File Number','Class'],as_index=True)['Seriousness','Violent'].sum()
grouped_file = grouped_SUcharge.groupby(['File Number']).sum()

#print( a.groupby('File Number')['Class'].apply(lambda x: "[%s]" % ', '.join(x)) )

grouped_codes = NewSUCharges.groupby('File Number')['Class'].apply(lambda x: "%s" % ','.join(x))

# Use dictionary to compare code class 
dict_code = {
        "A": 4,
        "B": 3,
        "C": 2,
        "M": 1,
        "GM": 1
        }

codes_list = list()
for fn, codes in zip(grouped_codes.index, grouped_codes):
#    print("fn: {} codes: {}".format(fn, codes))
    temp_fn_codes = list()
    for code in codes.split(','):
        #print("code: {} equals {}".format(code, dict_code[code]))
        temp_fn_codes.append(dict_code[code])
    highest_class = max(temp_fn_codes)
    for key, value in dict_code.items():
        if highest_class == value:
            highest_class = key
#    print("File Number: {} highest code: {}".format(fn, highest_code))

    codes_list.append(highest_class)
grouped_file.insert(2, "Highest Class", codes_list)

### Visualize DFA by Charge/Seriousness 


DFA_charge = grouped_file.merge(DFA1, left_index=True, right_on ='File Number', how='left')
DFA_charge = DFA_charge[['File Number','Highest Class','Seriousness','Violent','FailureToAppear','DFA']]

#DFA_charge.to_excel("DFA_charge.xlsx")

# Categorize file in Class and Serioussness level
serA1 = DFA_charge.where((DFA_charge['Highest Class'] == 'A') & (DFA_charge['Seriousness'] < 10)).dropna()
serB1 = DFA_charge.where((DFA_charge['Highest Class'] == 'B') & (DFA_charge['Seriousness'] < 10)).dropna()
serC1 = DFA_charge.where((DFA_charge['Highest Class'] == 'C') & (DFA_charge['Seriousness'] < 10)).dropna()
serM1 = DFA_charge.where((DFA_charge['Highest Class'] == 'M') & (DFA_charge['Seriousness'] < 10)).dropna()

serA2 = DFA_charge.where((DFA_charge['Highest Class'] == 'A') & (DFA_charge['Seriousness'] >= 10) & (DFA_charge['Seriousness'] <= 20)).dropna()
serB2 = DFA_charge.where((DFA_charge['Highest Class'] == 'B') & (DFA_charge['Seriousness'] >= 10) & (DFA_charge['Seriousness'] <= 20)).dropna()
serC2 = DFA_charge.where((DFA_charge['Highest Class'] == 'C') & (DFA_charge['Seriousness'] >= 10) & (DFA_charge['Seriousness'] <= 20)).dropna()
serM2 = DFA_charge.where((DFA_charge['Highest Class'] == 'M') & (DFA_charge['Seriousness'] >= 10) & (DFA_charge['Seriousness'] <= 20)).dropna()

serA3 = DFA_charge.where((DFA_charge['Highest Class'] == 'A') & (DFA_charge['Seriousness'] > 20)).dropna()
serB3 = DFA_charge.where((DFA_charge['Highest Class'] == 'B') & (DFA_charge['Seriousness'] > 20)).dropna()
serC3 = DFA_charge.where((DFA_charge['Highest Class'] == 'C') & (DFA_charge['Seriousness'] > 20)).dropna()
serM3 = DFA_charge.where((DFA_charge['Highest Class'] == 'M') & (DFA_charge['Seriousness'] > 20)).dropna()
           
# DFA in each category of Class and Seriousness level
DFAserA1 = round((serA1['DFA'].sum()/serA1['DFA'].count())*100,1)
DFAserB1 = round((serB1['DFA'].sum()/serB1['DFA'].count())*100,1)
DFAserC1 = round((serC1['DFA'].sum()/serC1['DFA'].count())*100,1)
DFAserM1 = round((serM1['DFA'].sum()/serM1['DFA'].count())*100,1)

DFAserA2 = round((serA2['DFA'].sum()/serA2['DFA'].count())*100,1)
DFAserB2 = round((serB2['DFA'].sum()/serB2['DFA'].count())*100,1)
DFAserC2 = round((serC2['DFA'].sum()/serC2['DFA'].count())*100,1)
DFAserM2 = round((serM2['DFA'].sum()/serM2['DFA'].count())*100,1)

DFAserA3 = round((serA3['DFA'].sum()/serA3['DFA'].count())*100,1)
DFAserB3 = round((serB3['DFA'].sum()/serB3['DFA'].count())*100,1)
DFAserC3 = round((serC3['DFA'].sum()/serC3['DFA'].count())*100,1)
DFAserM3 = round((serM3['DFA'].sum()/serM3['DFA'].count())*100,1)


DFA_ser1 = [DFAserA1, DFAserB1, DFAserC1, DFAserM1]
DFA_ser2 = [DFAserA2, DFAserB2, DFAserC2, 0]
DFA_ser3 = [DFAserA3, DFAserB3, DFAserC2, 0]

labels = ['A', 'B', 'C', 'M']
x = np.arange(len(labels))  # the label locations 
width = 0.30  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x, DFA_ser1, width, label='Seriousness < 10')
rects2 = ax.bar(x + width, DFA_ser2, width, label='Seriousness 10-20')
rects3 = ax.bar(x + width*2, DFA_ser3, width, label='Seriousness > 20')

# Add some text for labels, title and custom x-axis tick labels
ax.set_ylabel('Percentage')
ax.set_title('Failure to Appear by Charge Class and Seriousness Level')
ax.set_xticks(x+width)
ax.set_xticklabels(labels)
ax.legend()

def autolabel(rects):
    #Attach a text label above each bar in *rects*, displaying its height 
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

fig.tight_layout()

plt.show()

# In[ ]:

### Visualize DFA by Violent level in each Class/Seriousness category 

for v in range(0,2):
    if v == 0:
        serA1v = DFA_charge.where((DFA_charge['Highest Class'] == 'A') & (DFA_charge['Seriousness'] < 10) & (DFA_charge['Violent'] == v)).dropna()
        serB1v = DFA_charge.where((DFA_charge['Highest Class'] == 'B') & (DFA_charge['Seriousness'] < 10) & (DFA_charge['Violent'] == v)).dropna()
        serC1v = DFA_charge.where((DFA_charge['Highest Class'] == 'C') & (DFA_charge['Seriousness'] < 10) & (DFA_charge['Violent'] == v)).dropna()
        serM1v = DFA_charge.where((DFA_charge['Highest Class'] == 'M') & (DFA_charge['Seriousness'] < 10) & (DFA_charge['Violent'] == v)).dropna()
        
        serA2v = DFA_charge.where((DFA_charge['Highest Class'] == 'A') & (DFA_charge['Seriousness'] >= 10) & (DFA_charge['Seriousness'] <= 20) & (DFA_charge['Violent'] == v)).dropna()
        serB2v = DFA_charge.where((DFA_charge['Highest Class'] == 'B') & (DFA_charge['Seriousness'] >= 10) & (DFA_charge['Seriousness'] <= 20) & (DFA_charge['Violent'] == v)).dropna()
        serC2v = DFA_charge.where((DFA_charge['Highest Class'] == 'C') & (DFA_charge['Seriousness'] >= 10) & (DFA_charge['Seriousness'] <= 20) & (DFA_charge['Violent'] == v)).dropna()
        serM2v = DFA_charge.where((DFA_charge['Highest Class'] == 'M') & (DFA_charge['Seriousness'] >= 10) & (DFA_charge['Seriousness'] <= 20) & (DFA_charge['Violent'] == v)).dropna()
        
        serA3v = DFA_charge.where((DFA_charge['Highest Class'] == 'A') & (DFA_charge['Seriousness'] > 20) & (DFA_charge['Violent'] == v)).dropna()
        serB3v = DFA_charge.where((DFA_charge['Highest Class'] == 'B') & (DFA_charge['Seriousness'] > 20) & (DFA_charge['Violent'] == v)).dropna()
        serC3v = DFA_charge.where((DFA_charge['Highest Class'] == 'C') & (DFA_charge['Seriousness'] > 20) & (DFA_charge['Violent'] == v)).dropna()
        serM3v = DFA_charge.where((DFA_charge['Highest Class'] == 'M') & (DFA_charge['Seriousness'] > 20) & (DFA_charge['Violent'] == v)).dropna()
    else:
        serA1v = DFA_charge.where((DFA_charge['Highest Class'] == 'A') & (DFA_charge['Seriousness'] < 10) & (DFA_charge['Violent'] >= v)).dropna()
        serB1v = DFA_charge.where((DFA_charge['Highest Class'] == 'B') & (DFA_charge['Seriousness'] < 10) & (DFA_charge['Violent'] >= v)).dropna()
        serC1v = DFA_charge.where((DFA_charge['Highest Class'] == 'C') & (DFA_charge['Seriousness'] < 10) & (DFA_charge['Violent'] >= v)).dropna()
        serM1v = DFA_charge.where((DFA_charge['Highest Class'] == 'M') & (DFA_charge['Seriousness'] < 10) & (DFA_charge['Violent'] >= v)).dropna()
        
        serA2v = DFA_charge.where((DFA_charge['Highest Class'] == 'A') & (DFA_charge['Seriousness'] >= 10) & (DFA_charge['Seriousness'] <= 20) & (DFA_charge['Violent'] >= v)).dropna()
        serB2v = DFA_charge.where((DFA_charge['Highest Class'] == 'B') & (DFA_charge['Seriousness'] >= 10) & (DFA_charge['Seriousness'] <= 20) & (DFA_charge['Violent'] >= v)).dropna()
        serC2v = DFA_charge.where((DFA_charge['Highest Class'] == 'C') & (DFA_charge['Seriousness'] >= 10) & (DFA_charge['Seriousness'] <= 20) & (DFA_charge['Violent'] >= v)).dropna()
        serM2v = DFA_charge.where((DFA_charge['Highest Class'] == 'M') & (DFA_charge['Seriousness'] >= 10) & (DFA_charge['Seriousness'] <= 20) & (DFA_charge['Violent'] >= v)).dropna()
        
        serA3v = DFA_charge.where((DFA_charge['Highest Class'] == 'A') & (DFA_charge['Seriousness'] > 20) & (DFA_charge['Violent'] >= v)).dropna()
        serB3v = DFA_charge.where((DFA_charge['Highest Class'] == 'B') & (DFA_charge['Seriousness'] > 20) & (DFA_charge['Violent'] >= v)).dropna()
        serC3v = DFA_charge.where((DFA_charge['Highest Class'] == 'C') & (DFA_charge['Seriousness'] > 20) & (DFA_charge['Violent'] >= v)).dropna()
        serM3v = DFA_charge.where((DFA_charge['Highest Class'] == 'M') & (DFA_charge['Seriousness'] > 20) & (DFA_charge['Violent'] >= v)).dropna()
    
    # DFA in each category of Class and Seriousness level
    DFAserA1v = round((serA1v['DFA'].sum()/serA1v['DFA'].count())*100,1)
    DFAserB1v = round((serB1v['DFA'].sum()/serB1v['DFA'].count())*100,1)
    DFAserC1v = round((serC1v['DFA'].sum()/serC1v['DFA'].count())*100,1)
    DFAserM1v = round((serM1v['DFA'].sum()/serM1v['DFA'].count())*100,1)
    
    DFAserA2v = round((serA2v['DFA'].sum()/serA2v['DFA'].count())*100,1)
    DFAserB2v = round((serB2v['DFA'].sum()/serB2v['DFA'].count())*100,1)
    DFAserC2v = round((serC2v['DFA'].sum()/serC2v['DFA'].count())*100,1)
    DFAserM2v = round((serM2v['DFA'].sum()/serM2v['DFA'].count())*100,1)
    
    DFAserA3v = round((serA3v['DFA'].sum()/serA3v['DFA'].count())*100,1)
    DFAserB3v = round((serB3v['DFA'].sum()/serB3v['DFA'].count())*100,1)
    DFAserC3v = round((serC3v['DFA'].sum()/serC3v['DFA'].count())*100,1)
    DFAserM3v = round((serM3v['DFA'].sum()/serM3v['DFA'].count())*100,1)
    
    DFA_ser1v = [DFAserA1v, DFAserB1v, DFAserC1v, DFAserM1v]
    DFA_ser2v = [DFAserA2v, DFAserB2v, DFAserC2v, 0]
    DFA_ser3v = [DFAserA3v, DFAserB3v, DFAserC3v, 0]
    
    labels = ['A', 'B', 'C', 'M']
    x1 = np.arange(len(labels))  # the label locations 
    width = 0.30  # the width of the bars
    
    fig, ax1 = plt.subplots()
    rects1v = ax1.bar(x1, DFA_ser1v, width, label='Seriousness < 10')
    rects2v = ax1.bar(x1 + width, DFA_ser2v, width, label='Seriousness 10-20')
    rects3v = ax1.bar(x1 + width*2, DFA_ser3v, width, label='Seriousness > 20')
    
    # Add some text for labels, title and custom x-axis tick labels
    ax1.set_ylabel('Percentage')
    if v == 0:
        ax1.set_title('Failure to Appear by Non-Violent Charges and Seriousness')
    else:
        ax1.set_title('Failure to Appear by Violent Charges and Seriousness')
        
    ax1.set_xticks(x1+width)
    ax1.set_xticklabels(labels)
    if v == 0:
        ax1.legend(prop={"size":7.5})
    else:
        ax1.legend()
    
    def autolabel(rects):
        #Attach a text label above each bar in *rects*, displaying its height 
        for rect in rects:
            height = rect.get_height()
            ax1.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    autolabel(rects1v)
    autolabel(rects2v)
    autolabel(rects3v)
    
    fig.tight_layout()
    
    plt.show()
        
        
# In[ ]:
# ### EDA 1.7: DFA rate based on previous failure to appear of defendants
    
DFA1_previousDFA = DFA1[['File Number','Defendant ID','Event Enter Date','DFA']].sort_values(by=['Defendant ID', 'Event Enter Date'])

grouped_DFA_defe = DFA1_previousDFA.groupby('Defendant ID')['DFA'].apply(list)

previous_dfa_list = list()
for defe, dfa  in zip(grouped_DFA_defe.index, grouped_DFA_defe):
    new_file = True
    previous_dfa = False
    for d in dfa:
        if d == 1:
            if new_file is True: 
                previous_dfa_list.append(0)
            else:
                if previous_dfa is False:
                    previous_dfa_list.append(0)
                else:
                    previous_dfa_list.append(1)
            previous_dfa = True
        else:
            if previous_dfa is True:
                previous_dfa_list.append(1)
            else:
                previous_dfa_list.append(0)
            previous_dfa = False
        new_file = False
        
DFA1_previousDFA['Previous DFA'] = previous_dfa_list    

DFA1_previousDFA.to_excel('DFA1_previousDFA.xlsx')

### Visualize DFA rate Previous DFA:

DFA1_previousDFA['preDFA'] = np.where(DFA1_previousDFA['Previous DFA'] == 0, 'No Previous DFA', 'Previous DFA')
                                            
failureRatePreDFA = DFA1_previousDFA.groupby(['preDFA'])['DFA'].sum()/DFA1_previousDFA.groupby(['preDFA'])['DFA'].count()
failureRatePreDFA = failureRatePreDFA.apply(lambda x: round(x*100,2))
print(failureRatePreDFA)

plt.bar(failureRatePreDFA.index,failureRatePreDFA)
plt.title('Likelihood to Fail to Appear')
plt.ylabel('Percentage')
plt.show()

# In[ ]:
# ###  1.8 DFA rate by Case Types:

DFA1['Case Types'] = DFA1['Case Types'].str.split(', ').tolist() #split to list


DFA1['Case No 1'] = DFA1['Case Types'].str[0] #get the 1st case
DFA1['Case No 2'] = DFA1['Case Types'].str[1] #get the 2nd case
DFA1['Case No 3'] = DFA1['Case Types'].str[2] #get the 3rd case
DFA1['Case No 4'] = DFA1['Case Types'].str[3]
DFA1['Case No 5'] = DFA1['Case Types'].str[4]
DFA1['Case No 6'] = DFA1['Case Types'].str[5]


cases = [c for c in DFA1 if c.startswith('Case No')]
DFACases = pd.melt(DFA1, id_vars='File Number', value_vars=cases, value_name='Cases')

DFACases['Cases'].value_counts()[0:20]


#Case Types Categories
DFACases['CaseCat'] = np.where(DFACases['Cases'] == 'VUCSA','VUCSA',
                               np.where(DFACases['Cases'] == 'Domestic violence', 'Domestic violence',
                                np.where(DFACases['Cases'] == 'eDiscovery', 'eDiscovery',
                                np.where(DFACases['Cases'] == 'Intimate Partner Violence', 'Intimate Partner Violence',
                                np.where(DFACases['Cases'] == 'Car Theft Initiative', 'Car Theft Initiative', 'Others')))))

DFACaseTypes = pd.merge(DFA1, DFACases, on = 'File Number', how = 'left')
DFACaseTypes.columns


failureRateCaseType = DFACaseTypes.groupby(['File Number','CaseCat'])['DFA'].sum()/DFACaseTypes.groupby(['File Number','CaseCat'])['DFA'].count()
failureRateCaseType = failureRateCaseType.apply(lambda x: round(x*100,2))
failureRateCaseType

temp = failureRateCaseType.unstack('CaseCat').mean()


plt.bar(temp.index,temp)
plt.title('Failure to Appear by Case Types')
plt.ylabel('Percentage')
plt.xticks(rotation=90)
plt.show()

# In[ ]:

# ### 1.9. DFA rate by Custody Status

DFAcs = caseEventsSUTrain.groupby(['Defendant_Event_Status'])['FailureToAppear'].sum()/caseEventsSUTrain.groupby(['Defendant_Event_Status'])['FailureToAppear'].count()   
DFAcs = DFAcs.apply(lambda x: round(x*100,2))
DFAcs

DFAcs.plot.bar(x=None)
plt.title('Failure to Appear by Custody Status')
plt.show()


# In[ ]:

# ### 1.10. DFA rate by Case Setting:

caseSettingCodes = ['HRCSGP','HRCSNWK','HRCSNWS','HRCSWK','HRCSWS']
uDID = temp['Defendant ID'].unique()
temp= caseEventsSUTrain[caseEventsSUTrain['Defendant ID'] == 389357]
uniqueFile = caseEventsSUTrain['File Number'].unique()
temp1 = pd.merge(left = temp, right = CaseEventKey1, left_on = 'Event_Code',right_on = 'Hearing Code', how = 'inner')

def isBefore(date, dateArray):
    #if list is empty
    if len(dateArray) == 0:
        return int(True)
    #return if the date is before or not
    #1 = true and false = 0 
    return int(date < min(dateArray))
    
def isAfter(date, dateArray):
    #if list is empty
    if len(dateArray) == 0:
        return int(False)
    #return if the date is before or not
    #1 = true and false = 0 
    return int(date > min(dateArray))

uniqueFile = temp1['File Number'].unique()
len(uniqueFile)

caseSettingsDates = temp1[(temp1['Event_Code'].isin(caseSettingCodes)) & (temp1['File Number'] == uf)]['Event_Enter_Date']
temp1['DFABefore'] = temp1.apply(lambda row: isBefore(row['Event_Enter_Date'],temp1[(temp1['Event_Code'].isin(caseSettingCodes)) & (temp1['File Number'] == row['File Number'])]['Event_Enter_Date']) 
                                 if row['Event_Code'] == 'DEFFTA' else 0, axis=1)
temp1['DFAAfter'] = temp1.apply(lambda row: isAfter(row['Event_Enter_Date'],temp1[(temp1['Event_Code'].isin(caseSettingCodes)) & (temp1['File Number'] == row['File Number'])]['Event_Enter_Date']) 
                                     if row['Event_Code'] == 'DEFFTA' else 0, axis=1)
temp1

start = time.time()
caseEventsSUTrain['DFABefore'] = caseEventsSUTrain.apply(lambda row: isBefore(row['Event_Enter_Date'],caseEventsSUTrain[(caseEventsSUTrain['Event_Code'].isin(caseSettingCodes)) & (caseEventsSUTrain['File Number'] == row['File Number'])]['Event_Enter_Date']) 
                                 if row['Event_Code'] == 'DEFFTA' else 0, axis=1)
caseEventsSUTrain['DFAAfter'] = caseEventsSUTrain.apply(lambda row: isAfter(row['Event_Enter_Date'],caseEventsSUTrain[(caseEventsSUTrain['Event_Code'].isin(caseSettingCodes)) & (caseEventsSUTrain['File Number'] == row['File Number'])]['Event_Enter_Date']) 
                                     if row['Event_Code'] == 'DEFFTA' else 0, axis=1)
elapsed_time_fl = (time.time() - start)
print(elapsed_time_fl)


FiledCasesSU = data1['FiledCases SU']
FiledCasesSUTrain = FiledCasesSU[FiledCasesSU['Defendant ID'].isin(dID_train)]
caseEventsGroup = caseEventsSUTrain.groupby(['File Number'])['DFABefore','DFAAfter'].sum()
FiledCasesSUTrain1 = pd.merge(left = FiledCasesSUTrain, right = caseEventsGroup, left_on = 'File Number',right_on = 'File Number',how = 'left')
FiledCasesSUTrain1.columns

caseSettingRateBefore = FiledCasesSUTrain1['DFAAfter'].sum()/FiledCasesSUTrain1['DFAAfter'].count()
caseSettingRateBefore

print(caseEventsGroup['DFABefore'].sum() + caseEventsGroup['DFAAfter'].sum())
print(caseEventsSUTrain.loc[caseEventsSUTrain['Event_Code'] == 'DEFFTA'].shape)
before = round(FiledCasesSUTrain1['DFABefore'].sum()/FiledCasesSUTrain1['DFABefore'].count(),2)*100
after = round(FiledCasesSUTrain1['DFAAfter'].sum()/FiledCasesSUTrain1['DFAAfter'].count(),2)*100


x = ['DFABefore','DFAAfter']
y = [before,after]
fig, ax = plt.subplots()
ax.bar(x,y)
ax.set_title('DFA Rate Before and After Case Setting')
ax.set_ylabel('Percentage')


# In[ ]:

#Time to disposition

#merge to get rankings
TTD = pd.merge(SUCharges, DispositionKey, on = 'Disposition Code', how = 'left')
TTD = TTD.dropna(subset=['Ranking'])
TTD = TTD.dropna(subset=['Disposition Date'])

TTD
TTD.columns

#convert to datetime
TTD['Disposition Date'] = pd.to_datetime(TTD['Disposition Date'])
TTD['Filing Date'] = pd.to_datetime(TTD['Filing Date'])

#get time to disposition
TTD['Time to Disposition'] = TTD['Disposition Date'] - TTD['Filing Date']


##if multiple disposition dates, drop duplicates with ranking greater than 1
TTD[TTD.Ranking.gt(1) & ~(TTD.duplicated(['File Number']))]



#merge TTD with FiledCases
TTDCase = pd.merge(left = FiledCasesSUTrain,right = TTD, left_on = 'File Number',right_on = 'File Number',how = 'left')
TTDCase.columns

#TTD Gender
TTDGender = TTDCase.groupby(['File Number','Gender'])['Time to Disposition'].sum()

temp = TTDGender.unstack('Gender').mean()
temp = temp.astype('timedelta64[D]')

plt.bar(temp.index,temp)
plt.title('Time to Disposition by Gender')
plt.ylabel('Days')
plt.show()

#TTD by Age group
bins= [0,19,30,40,50,60,150]
labels = ['<19','20-29','30-39','40-49','50-59','60+']
TTDCase['AgeGroup'] = pd.cut(TTDCase['Age'], bins=bins, labels=labels, right=False)



TTDAge = TTDCase.groupby(['File Number','AgeGroup'])['Time to Disposition'].sum()

temp = TTDAge.unstack('AgeGroup').mean()
temp = temp.astype('timedelta64[D]')

plt.bar(temp.index,temp)
plt.title('Time to Disposition by Age Group')
plt.ylabel('Days')
plt.show()

# =============================================================================
# #TTD Custody Status
# TTDCust = pd.merge(left = FiledCasesSUTrain,right = TTD, left_on = 'File Number',right_on = 'File Number',how = 'left')
# 
# =============================================================================



#TTD Drug Court
caseEventsSUTrain['Drug'] = caseEventsSUTrain['Event_Code'].apply(lambda x: 1 
                                    if x == 'HRARDR' or x == 'HRBWSC' or x == 'HRDR1' 
                                    or x == 'HRDR2' or x == 'HRDRC' else 0)


Drug = caseEventsSUTrain.groupby('File Number')['Drug'].sum()
Drug


TTDDrug = pd.merge(left = TTDCase, right = Drug, left_on = 'File Number',right_on = 'File Number',how = 'left')
TTDDrug['Drug Court'] = np.where(TTDDrug['Drug'] > 0, 1, 0)

TTDDrug['Time to Disposition'] = TTDDrug['Time to Disposition'].astype('timedelta64[D]')

TTDDrugCourt = TTDDrug.groupby(['File Number','Drug Court'])['Time to Disposition'].sum()

temp = TTDDrugCourt.unstack('Drug Court').mean()
temp = temp.astype('timedelta64[D]')

bars = ('Non-drug Court', 'Drug Court')
plt.bar(bars,temp)
plt.title('Time to Disposition by Drug Court Cases')
plt.ylabel('Days')
plt.show()

#TTD Venue:

TTDCase['Venue_Dummies'] = ""

#Add in the dummies based on Venue:
TTDCase['Venue_Dummies'] = np.where(TTDCase['Venue'] == 'SEA', 'Seattle Venue', 
                                 np.where(TTDCase['Venue'] == 'KNT', 'Kent Venue', 'Other Venues'))
TTDCase['Venue_Dummies'].value_counts()

TTDVenue = TTDCase.groupby(['File Number','Venue_Dummies'])['Time to Disposition'].sum()/TTDCase.groupby(['File Number','Venue_Dummies'])['Time to Disposition'].count()


temp = TTDVenue.unstack('Venue_Dummies').mean()
temp = temp.astype('timedelta64[D]')
plt.bar(temp.index,temp)
plt.title('Time to Disposition by Venue')
plt.ylabel('Days')
plt.show()

#TTD Police Depratment:

TTDCase['Police_Dummies'] = ""

#Add in the dummies based on Police Agency:
TTDCase['Police_Dummies'] = np.where(TTDCase['Police Agency'] == 'Seattle Police Department','Agent1',
                                     np.where(TTDCase['Police Agency'] == "King County Sheriff's Office", 'Agent2',
                                     np.where(TTDCase['Police Agency'] == 'Auburn Police Department', 'Agent3',
                                     np.where(TTDCase['Police Agency'] == 'Kent Police Department', 'Agent4',
                                     np.where(TTDCase['Police Agency'] == 'Federal Way Police Department', 'Agent5',
                                     np.where(TTDCase['Police Agency'] == 'Renton Police Department', 'Agent6',
                                     np.where(TTDCase['Police Agency'] == 'Bellevue Police Department', 'Agent7', 'Others')))))))

#DFA by Police Agency:

TTDPolicia = TTDCase.groupby(['File Number','Police_Dummies'])['Time to Disposition'].sum()/TTDCase.groupby(['File Number','Police_Dummies'])['Time to Disposition'].count()

temp = TTDPolicia.unstack('Police_Dummies').mean()
temp = temp.astype('timedelta64[D]')
plt.bar(temp.index,temp)
plt.title('Time to Disposition by Police Agency')
plt.ylabel('Days')
plt.show()

#TTD by Criminal Record and COnvcition Level:

#Group "AM" together:
SUCrimHist2Train['Convic_Dummies'] = np.where(SUCrimHist2Train['ConvictionLevel'] == 'AF', 'AF',
                                              np.where(SUCrimHist2Train['ConvictionLevel'] == 'JM', 'JM',
                                              np.where(SUCrimHist2Train['ConvictionLevel'] == 'JF','JF','AM')))
#Merge Crimhist and DFA1:
TTD2= SUCrimHist2Train.merge(right = TTDCase.drop_duplicates(subset=['Defendant ID']), left_on = 'DefendantID', right_on = 'Defendant ID', how ='left')
TTD2['Time to Disposition'].value_counts()
TTD2.count()

#Graph the DFA by conviction level
DFACon = TTD2.groupby(['Defendant ID','Convic_Dummies'])['Time to Disposition'].sum()/TTD2.groupby(['Defendant ID','Convic_Dummies'])['Time to Disposition'].count()
DFACon = failureRateCL.apply(lambda x: round(x*100,2))

temp = failureRateCL.unstack('Convic_Dummies').mean()
plt.bar(temp.index,temp)
plt.title('Failure to Appear by Conviction Level')
plt.ylabel('Percentage')
plt.show()
DFA2['Convic_Dummies'].value_counts()
#Show mising values in DFA2:
miss.matrix(DFA2)

#Make the colum Historical offense in Year:
DFA2['OffenseDate'] = pd.to_datetime(DFA2['OffenseDate'],errors='coerce')
DFA2 = DFA2.dropna(subset = ['OffenseDate']) #Drop missing values of Offense date- only 4%
DFA2['Historical Offense'] = DFA2['OffenseDate'].dt.date.apply( calculateAge)

#Plot the chart with both Criminal record and Convcition Level

conAM1 = DFA1.where((DFA2['Convic_Dummies'] == 'AM') & (DFA2['Historical Offense'] <= 5)).dropna()
conAF1 = DFA1.where((DFA2['Convic_Dummies'] == 'AF') & (DFA2['Historical Offense'] <= 5)).dropna()
conJM1 = DFA1.where((DFA2['Convic_Dummies'] == 'JM') & (DFA2['Historical Offense'] <= 5)).dropna()
conJF1 = DFA1.where((DFA2['Convic_Dummies'] == 'JF') & (DFA2['Historical Offense'] <= 5)).dropna()

conAM2 = DFA1.where((DFA2['Convic_Dummies'] == 'AM') & (DFA2['Historical Offense'] > 5)).dropna()
conAF2 = DFA1.where((DFA2['Convic_Dummies'] == 'AF') & (DFA2['Historical Offense'] > 5)).dropna()
conJM2 = DFA1.where((DFA2['Convic_Dummies'] == 'JM') & (DFA2['Historical Offense'] < 5)).dropna()
conJF2 = DFA1.where((DFA2['Convic_Dummies'] == 'JF') & (DFA2['Historical Offense'] > 5)).dropna()

DFAcon1 = round((conAM1['DFA'].sum()/conAM1['DFA'].count())*100,1)
DFAcon2 = round((conAF1['DFA'].sum()/conAF1['DFA'].count())*100,1)
DFAcon3 = round((conJM1['DFA'].sum()/conJM1['DFA'].count())*100,1)
DFAcon4 = round((conJF1['DFA'].sum()/conJF1['DFA'].count())*100,1)

DFAcon5 = round((conAM2['DFA'].sum()/conAM2['DFA'].count())*100,1)
DFAcon6 = round((conAF2['DFA'].sum()/conAF2['DFA'].count())*100,1)
DFAcon7 = round((conJM2['DFA'].sum()/conJM2['DFA'].count())*100,1)
DFAcon8 = round((conJF2['DFA'].sum()/conJF2['DFA'].count())*100,1)

DFA_con1 = [DFAcon1, DFAcon2, DFAcon3, DFAcon4]
DFA_con2 = [DFAcon5, DFAcon6, DFAcon7, 0]

labels = ['AM', 'AF', 'JM', 'JF']
x = np.arange(len(labels))  # the label locations 
width = 0.30  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x, DFA_con1, width, label='Recent')
rects2 = ax.bar(x + width, DFA_con2, width, label='Old')


# Add some text for labels, title and custom x-axis tick labels
ax.set_ylabel('Percentage')
ax.set_title('Failure to Appear by Criminal Record and Conviction Level')
ax.set_xticks(x + 0.5*width)
ax.set_xticklabels(labels)
ax.legend(handletextpad = 0.1, borderaxespad = 0.1)

def autolabel(rects):
    #Attach a text label above each bar in *rects*, displaying its height 
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)


fig.tight_layout()

plt.show()