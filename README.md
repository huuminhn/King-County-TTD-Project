<img src="hearing_court.png?raw=true"/>

## Forecasting the Defendant Failure to Appear rate and the Time to Disposition of a case:

This is a team project that research on Failure to Appear rate of defendants and time to disposition of a case in a hearing court.    
Due to privacy, the original dataset and code are not allowed to be shared. 

---
#### The background of the project:
Before a case is presented in front of a court, a hearing would be scheduled to justify the legitimation of the case, whether the defendant has admissibility of evidence or dismiss the case without further trials. However, for various reasons, many defendants failed to appear for the hearing. This project aims to analyze factors that potentially impact the probability of failure to appear of defendants, along with exploration of the time-to-disposition of hearing cases.  
The data used in this project contain 6 CSV files, 1.5 million rows and over 30 variables collected from King County Prosecuting Attorney Office.  

#### Data cleaning, EDA and modeling: 
1. Data cleaning:
- Cleaned, merged, manipulated, and aggregated data  via Pandas and Numpy.  
- Outter join various CSV files together to form a single CSV file for the whole team to work on with efficiency.  
- Fixed cells with multiple data by re-categorizing.  
- Fill in emty cell (Null) with data aggregated from other CSV files.  
2. EDA: 
- Constructed a correlation matrix to eliminate multicollinearity.
- Applied the ANOVA test and Holdout Method to came up with potential independent variables for model building.  
3. Modeling: 
- Developed statistical models (logistic regression) as a benchmark for machine learning models (decision tree, SVN, KNN, and neural network).  
- Variable selected in the Logistic Regression model are used as a guideline for variables in other models.
- Perform trimming techniques on machine learning models to obtain the optimal confusion matrixes. 

#### Solution:   
- The EDA step increased the performance of model by 38%.    
- The accuracy rate of prediction is up to 68% (with the highest belong to the Neutral Network model).  
- The precision of prediction is up to 65% ( with the highest belong to the Linear Regression Model).
- The sensitivity of prediction is up to 66% ( with the highest belong to the KNN Model). 
- The team’s research paper was later on submitted to the King County Prosecuting Attorney Office and being considered as a background for public policies adjustments. 
