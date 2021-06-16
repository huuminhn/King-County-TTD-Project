<img src="hearing_court.png?raw=true"/>

# Forecasting the Defendant Failure to Appear rate and the Time to Disposition of a case:

This is a team project that research on Failure to Appear rate of defendants and time to disposition of a case in a hearing court.    
Due to privacy, the original dataset and some part of the codes are not allowed to be shared. 

---
## The background of the project:
Before a case is presented in front of a court, a hearing would be scheduled to justify the legitimation of the case, whether the defendant has admissibility of evidence or dismiss the case without further trials. However, for various reasons, many defendants failed to appear for the hearing. This project aims to analyze factors that potentially impact the probability of failure to appear of defendants, along with exploration of the time-to-disposition of hearing cases.  
The data used in this project contains 6 CSV files, 1.5 million rows and over 30 variables collected from the King County Prosecuting Attorney Office.  

---

## 1. Data Cleaning:  

- Cleaned, merged, manipulated, and aggregated data  via Pandas and Numpy.  
- Future Engineering **6 CSV files** together to form a single CSV file for the whole team to work on with efficiency.  
- Fill in emty cell (Null) with data aggregated from other CSV files.  
- Fixed cells with multiple data by creating bins and spliting original variables to dummies variables.  
-   
## 2. EDA:  

- We process to explore the significant variables based on the impact and colinearity of a independent variable upon the dependent variable. We firstly calculate the DFA rate, then group the select x variable with the y variable and the DFA to see if that x variable is important for analyzation or not. Some example charts for this process are:  
<img src="dummies.png?raw=true"/>  
Those two charts on the right hand side indicate that Police Agencies, Charge Class and Seriousness are valuable for model building, whereas the other chart illustrates the insignificant of the Custody Status variable. Even though the differences of DFA rate among these four custody statuses are significant, the rate is still very low compared to the average rate, which is 44.19%. Thus, we excluded Custody Status as a predictor for DFA. Following the same logic, we create dummies and select valuable variables for the remaining x variables.  

- Constructed a correlation matrix to eliminate multicollinearity.  
<img src="corr.png?raw=true"/>  

- Applied the **ANOVA test and Holdout Method** to came up with potential independent variables for model building.  
  
## 3. Modeling (Logistic Regression, Decision Tree, SVM, KNN, Neutral Network):  
**a. Logistic Regression:**  
- Developed statistical models ( logistic regression) as a benchmark for machine learning models.   
- Variable selected in the Logistic Regression model are used as a guideline for variables in other models.The model can be summarized as below:  
<img src="logistic.png?raw=true"/>  

**b. Decision Tree:**  
- Decision Tree is one of the supervised machine learning algorithms and classification techniques to induce a decision tree from training data. A decision tree is a predictive model mapping observation about an item to conclude about its target value (Robert Nisbet, Gary Miner, John Elder, 2018).    
- Firstly, we applied some pre-prunning techniques to select the best parameters for the decision tree. The method can be illustrated as below:  
<img src="tree_prun.png?raw=true"/>  

- Based on the result of the graphs, we set max_depth = 13 for better interpretation, min_sample_leaf at 1000 (for the tree to expand more freely compared to that at 2000), and min_sample_split = 3000. As a result, here is our Decision Tree:  
<img src="tree.png?raw=true"/>  

**c. SVM, KNN and Neutral Network:**  

- **SVM**: We use significant variables from the Logistic Regression and Decision Tree models, then we test models with 3 kernels: Radial Basis Function (RBF), Linear, and Polynomial (degree = 3), then choose the one with the best result, which turn out to be the RBF kernel.  
- **KNN**: We applied pre-prunning technique to find the optimal K points, with variables suggestion coming from the previous 2 benchmark models.  
<img src="KNN.png?raw=true"/>  

- **Neutral Network**: Using the same chart plotting technique to observe the impact of parameters upon the model, we applied pre-prunning technique to come up with the best model.   
   
**d. Model Comparision:**  


#### Solution:   
- The EDA step increased the performance of model by 38%.    
- The accuracy rate of prediction is up to 68% (with the highest belong to the Neutral Network model).  
- The precision of prediction is up to 65% ( with the highest belong to the Linear Regression Model).
- The sensitivity of prediction is up to 66% ( with the highest belong to the KNN Model). 
- The team’s research paper was later on submitted to the King County Prosecuting Attorney Office and being considered as a background for public policies adjustments. 
