#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Supress Warnings

import warnings
import pandas as pd
warnings.filterwarnings('ignore')


# In[2]:


# importing CarPrice_Assignment.csv

CarData= pd.read_csv("C:/Users/ASUS/Downloads/CarPrice_Assignment.csv")

CarData.head()


# In[3]:


# shape of the dataset
CarData.shape


# In[4]:


# descibing the dataset
CarData.info()


# In[5]:


CarData.describe()


# In[6]:


#Data Cleaning
#Removing/Imputing missing values


# In[7]:


# checking for null values column wise

CarData.isnull().sum()


# In[8]:


# checking for null values row wise

CarData.isnull().sum(axis=1)


# In[9]:


#Data Preparation


# In[10]:


# Extract the car's company's name from the variable 'CarName' into the variable 'CarCompany'

CarData['CarCompany'] = CarData.CarName.apply(lambda x: str(x.split(' ')[0]))

# dropping the variable 'Carname'

CarData.drop('CarName', axis=1, inplace=True)

CarData.head()


# In[11]:


#checking the unique values under "CarCompany"

CarData.CarCompany.unique()


# In[12]:


# replacing the mis-spelling with correct ones

CarData.CarCompany.replace('maxda','mazda',inplace=True)

CarData.CarCompany.replace('porcshce','porsche',inplace=True)

CarData.CarCompany.replace('toyouta','toyota',inplace=True)

CarData.CarCompany.replace(['vokswagen','vw'],'volkswagen',inplace=True)

CarData.CarCompany.replace('Nissan', 'nissan',inplace=True)

# again checking the unique values

CarData.CarCompany.unique()


# In[13]:


#checking the unique values under "fueltype"

CarData.fueltype.unique()


# In[14]:


#checking the unique values under "aspiration"

CarData.aspiration.unique()


# In[15]:


#checking the unique values under "doornumber"

CarData.doornumber.unique()


# In[16]:


#checking the unique values under "carbody"

CarData.carbody.unique()


# In[17]:


#checking the unique values under "drivewheel"

CarData.drivewheel.unique()


# In[18]:


#checking the unique values under "enginelocation"

CarData.enginelocation.unique()


# In[19]:


#checking the unique values under "enginetype"

CarData.enginetype.unique()


# In[20]:


#checking the unique values under "cylindernumber"

CarData.cylindernumber.unique()


# In[21]:


#checking the unique values under "fuelsystem"

CarData.fuelsystem.unique()


# In[22]:


#replacing 'mfi' with 'mpfi' in 'fuelsystem'

CarData.fuelsystem.replace('mfi','mpfi',inplace=True)

#again checking the unique values under "fuelsystem"

CarData.fuelsystem.unique()


# In[23]:


#Data Visualisation
#Numeric Variables


# In[24]:


# importing required libraries

import matplotlib.pyplot as plt
import seaborn as sns


# In[25]:


# pairplot of all numeric variables of dataset 'CarData'

sns.pairplot(CarData)
plt.show()


# In[26]:


# plot heatmap to check the correlation coefficients

plt.figure(figsize = (16, 10))
sns.heatmap(CarData.corr(), annot = True, cmap="YlGnBu")
plt.show()


# In[27]:


#Creation of Derived Variable


# In[28]:


# creating variable 'enginesize/horsepower'

CarData['enginesize/horsepower'] = CarData.enginesize/CarData.horsepower 


# In[29]:


# creating variable 'curbweight/enginesize'

CarData['curbweight/enginesize'] = CarData.curbweight/CarData.enginesize


# In[30]:


# creating variable 'carwidth/carlength'

CarData['carwidth/carlength'] = CarData.carwidth/CarData.carlength


# In[31]:


# creating variable 'highwaympg/citympg'

CarData['highwaympg/citympg'] = CarData.highwaympg/CarData.citympg


# In[32]:


# dropping the unncessary variables

CarData.drop(['enginesize', 'curbweight', 'horsepower', 'carwidth', 'carlength', 'highwaympg', 'citympg', 'car_ID'], 
             axis=1, inplace=True)

# checking the dataframe

CarData.head()


# In[33]:


#Categorical Variable


# In[34]:


#mapping in 'symboling'

CarData.symboling = CarData.symboling.map({-3: 'safe', -2: 'safe', -1: 'safe', 
                                           0: 'moderate', 1: 'moderate', 2:'risky', 3:'risky'})

#checking the 'symboling' column

CarData.symboling.head()


# In[35]:


# Visualising Categorical variables through boxplots

plt.figure(figsize=(20, 16))
plt.subplot(3,4,1)
sns.boxplot(x = 'symboling', y = 'price', data = CarData)
plt.subplot(3,4,2)
sns.boxplot(x = 'fueltype', y = 'price', data = CarData)
plt.subplot(3,4,3)
sns.boxplot(x = 'aspiration', y = 'price', data = CarData)
plt.subplot(3,4,4)
sns.boxplot(x = 'doornumber', y = 'price', data = CarData)
plt.subplot(3,4,5)
sns.boxplot(x = 'carbody', y = 'price', data = CarData)
plt.subplot(3,4,6)
sns.boxplot(x = 'drivewheel', y = 'price', data = CarData)
plt.subplot(3,4,7)
sns.boxplot(x = 'enginelocation', y = 'price', data = CarData)
plt.subplot(3,4,8)
sns.boxplot(x = 'enginetype', y = 'price', data = CarData)
plt.subplot(3,4,9)
sns.boxplot(x = 'cylindernumber', y = 'price', data = CarData)
plt.subplot(3,4,10)
sns.boxplot(x = 'fuelsystem', y = 'price', data = CarData)

plt.show()


# In[36]:


# Visualising 'CarComapny' variable aginst dependent variable 'price' via boxplot

plt.figure(figsize=(20, 16))
sns.boxplot(x = 'CarCompany', y = 'price', data = CarData)
plt.show()


# In[37]:


#creating a dict 'company price' where key = car's company and value = median of their price

company_price = dict(CarData.groupby('CarCompany').price.median())
company_price


# In[38]:


# division in the buckets of low i.e. below 10000, medium i.e. range b/w 10000 and 20000 and high i.e. above 20000

for i in company_price.keys():
    
    if company_price[i] <= 10000:
        company_price[i]='low'
        
    elif (company_price[i] > 10000) & (company_price[i] < 20000):
        company_price[i]='med'
        
    else:
        company_price[i]='high'
        
company_price


# In[39]:


# mapping the company_price on the 'CarCompany' column in the dataset 

CarData.CarCompany = CarData.CarCompany.map(company_price)

# checking the column's unique values

CarData.CarCompany.unique()


# In[40]:


#Creation of Dummy Variables


# In[41]:


#creating dummy variable for catwegorical variables

CarData = pd.get_dummies(CarData)

#checking the dataset

CarData.head()


# In[42]:


#Splitting of data into training and testing sets


# In[43]:


# importing required library
from sklearn.model_selection import train_test_split

# We specify this so that the train and test data set always have the same rows, respectively

CarData_train, CarData_test = train_test_split(CarData, train_size = 0.7, random_state = 100)

# shape of the train and test sets
print(CarData_train.shape)
print(CarData_test.shape)


# In[44]:


#Rescaling the Features
#We will use MinMax scaling.


# In[45]:


#importing required function for scaling

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()


# In[46]:


# Applying scalar to all the numerical variables

num_vars = ['wheelbase', 'carheight', 'boreratio', 'stroke', 'compressionratio',
       'peakrpm', 'price', 'enginesize/horsepower', 'curbweight/enginesize',
       'carwidth/carlength', 'highwaympg/citympg']


CarData_train[num_vars] = scaler.fit_transform(CarData_train[num_vars])

CarData_train.head()


# In[47]:


#Dividing into X and Y sets for the model building


# In[48]:


y_train = CarData_train.pop('price')
X_train = CarData_train


# In[49]:


#Building model
#We will be using the LinearRegression function from SciKit Learn for its compatibility with RFE (which is a utility from sklearn)

#RFE
#Recursive feature elimination


# In[50]:


# Importing RFE and LinearRegression

from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression


# In[51]:


# Running RFE with the output number of the variable equal to 10

lm = LinearRegression()
lm.fit(X_train, y_train)

rfe = RFE(lm,n_features_to_select=10)             # running RFE
rfe = rfe.fit(X_train, y_train)


# In[52]:


list(zip(X_train.columns,rfe.support_,rfe.ranking_))


# In[53]:


col = X_train.columns[rfe.support_]
col


# In[54]:


X_train.columns[~rfe.support_]


# In[55]:


#Building model using statsmodel, for the detailed statistics


# In[56]:


# Creating X_test dataframe with RFE selected variables

X_train_rfe = X_train[col]


# In[57]:


# Adding a constant variable 

import statsmodels.api as sm  #importing required library

X_train_rfe = sm.add_constant(X_train_rfe)


# In[58]:


lm = sm.OLS(y_train,X_train_rfe).fit()   # Running the linear model


# In[59]:


# summary of the linear model

print(lm.summary())


# In[60]:


#VIF(Variance Inflation Factor)


# In[61]:


# Calculate the VIFs for the model

from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
X = X_train_rfe
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[62]:


# Rebuilding model w/o 'fueltype_diesel'

X_train_new = X_train_rfe.drop(['fueltype_diesel'], axis = 1)
# Adding a constant variable 

X_train_lm = sm.add_constant(X_train_new)

lm = sm.OLS(y_train,X_train_lm).fit()   # Running the linear model


# In[63]:


#summary of new linear model

print(lm.summary())


# In[64]:


# Calculate the VIFs for the model

vif = pd.DataFrame()
X = X_train_new
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[65]:


# Rebuilding model w/o 'fuelsystem_idi'

X_train_new = X_train_new.drop(['fuelsystem_idi'], axis = 1)
# Adding a constant variable 

X_train_lm = sm.add_constant(X_train_new)

lm = sm.OLS(y_train,X_train_lm).fit()   # Running the linear model

#summary of new linear model

print(lm.summary())


# In[66]:


# Calculate the VIFs for the model

vif = pd.DataFrame()
X = X_train_new
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[67]:


# Rebuilding model w/o 'compressionratio'

X_train_new = X_train_new.drop(['compressionratio'], axis = 1)

# Adding a constant variable 

X_train_lm = sm.add_constant(X_train_new)

lm = sm.OLS(y_train,X_train_lm).fit()   # Running the linear model

#summary of new linear model

print(lm.summary())


# In[68]:


# Calculate the VIFs for the model

vif = pd.DataFrame()
X = X_train_new
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[69]:


#Residual Analysis of the train data
#So, now to check if the error terms are also normally distributed. Plot the histogram of the error terms


# In[70]:


y_train_pred = lm.predict(X_train_lm)


# In[71]:


# Plot the histogram of the error terms
fig = plt.figure()
sns.distplot((y_train - y_train_pred), bins = 20)
fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 
plt.xlabel('Errors', fontsize = 18)                         # X-label
plt.show()


# In[72]:


#Making Predictions
#Applying the scaling on the test sets


# In[73]:


# Applying scalar to all the numerical variables

num_vars = ['wheelbase', 'carheight', 'boreratio', 'stroke', 'compressionratio',
       'peakrpm', 'price', 'enginesize/horsepower', 'curbweight/enginesize',
       'carwidth/carlength', 'highwaympg/citympg']


CarData_test[num_vars] = scaler.transform(CarData_test[num_vars])

CarData_test.head()


# In[74]:


#Dividing into X_test and y_test
y_test = CarData_test.pop('price')
X_test = CarData_test


# In[75]:


# Using the model to make predictions.
X_test = X_test[col]


# In[76]:


# Creating X_test_new dataframe by dropping variables from X_test
X_test_new = X_test.drop(['compressionratio', 'fuelsystem_idi', 'fueltype_diesel'], axis=1)

# Adding a constant variable 
X_test_new = sm.add_constant(X_test_new)


# In[77]:


# Making predictions

y_pred = lm.predict(X_test_new)


# In[78]:


#Model Evaluation


# In[79]:


# Plotting y_test and y_pred to understand the spread.
fig = plt.figure()
plt.scatter(y_test,y_pred)
fig.suptitle('y_test vs y_pred', fontsize=20)              # Plot heading 
plt.xlabel('y_test', fontsize=18)                          # X-label
plt.ylabel('y_pred', fontsize=16)                          # Y-label
plt.show()


# In[80]:


#r_squared value of test set
from sklearn.metrics import r2_score

r2_score(y_true=y_test, y_pred=y_pred)


# In[ ]:





# In[ ]:




