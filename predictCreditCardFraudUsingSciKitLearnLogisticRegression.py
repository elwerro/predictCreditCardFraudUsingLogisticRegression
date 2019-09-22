#!/usr/bin/env python
# coding: utf-8

# Dependencies

# In[1]:


# https://www.data-blogger.com/2017/06/15/fraud-detection-a-simple-machine-learning-approach/
#install pandas and scikit learn versions in Jupyter notebook for this demo from
#!pip install seaborn==0.9.0
get_ipython().system('pip install pandas==0.19.2 scikit-learn==0.18.1')


#import dependencies
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Import csv data into a pandas dataFrame

# In[2]:


data = pd.read_csv('creditcard.csv')


# Visualize the data

# In[3]:


print(data)


# In[4]:


print(data.describe())


# In[5]:


data.shape


# Each row in the data represents a transaction
# Separate the data into two parts, X and Y, 
# X data will contain all the features of the transaction
# Y data will contain the label(aka class): 0 if it is NOT fraud or 1 if it is fraud

# In[61]:


# Only use the 'Amount' and 'V1', ..., 'V28' features
features = ['Amount'] + ['V%d' % number for number in range(1, 29)]

# The target variable which we would like to predict, is the 'Class' variable
target = 'Class'

# Now create an X variable (containing the features) and an y variable (containing only the target variable)
X = data[features]
y = data[target]
num_of_frauds_before_filtering = 0
for value in y:
    if value == 1:
        num_of_frauds_before_filtering = num_of_frauds_before_filtering + 1
print(num_of_frauds_before_filtering)


# 

# In[25]:


def normalize(X):
    """
    Make the distribution of the values of each variable similar by subtracting the mean and by dividing by the standard deviation.
    """
    for feature in X.columns:
        X[feature] -= X[feature].mean()
        X[feature] /= X[feature].std()
    return X


# In[8]:


#Normalize after splitting data into training and testing when doing training and modeling
#Visualize data
#to maybe get rid of variables that does not matter
#or maybe get rid of transactions that are repeated too many times therefore increasing the fraud ratio in the data
#to make training data higher quality
X_data_visualize = data[features]


# In[9]:


X_data_visualize.shape


# In[10]:


y.shape


# In[11]:


x_merged_to_y = pd.concat([X_data_visualize, y], axis=1)
x_merged_to_y.shape


# In[12]:


print(x_merged_to_y)


# In[13]:


ax = sns.scatterplot(x="Amount",y="Class",hue="Class",data=x_merged_to_y)


# In[15]:


fig = plt.figure(figsize=(15,8))
fig.subplots_adjust(hspace=0.6, wspace=0.8)
for i in range(1, 9):
    plt.subplot(2, 4, i)
    sns.scatterplot(x="V"+str(i),y="Class",hue="Class",data=x_merged_to_y)


# In[16]:


fig = plt.figure(figsize=(15,8))
fig.subplots_adjust(hspace=0.6, wspace=0.8)
for i in range(1, 9):
    plt.subplot(2, 4, i)
    sns.scatterplot(x="V"+str(i+8),y="Class",hue="Class",data=x_merged_to_y)


# In[17]:


fig = plt.figure(figsize=(15,8))
fig.subplots_adjust(hspace=0.6, wspace=0.8)
for i in range(1, 9):
    plt.subplot(2, 4, i)
    sns.scatterplot(x="V"+str(i+16),y="Class",hue="Class",data=x_merged_to_y)


# In[18]:


fig = plt.figure(figsize=(15,4))
fig.subplots_adjust(hspace=0.6, wspace=0.8)
for i in range(1, 5):
    plt.subplot(1, 4, i)
    sns.scatterplot(x="V"+str(i+24),y="Class",hue="Class",data=x_merged_to_y)


# In[62]:


#After visualizing the data, 
#I notice that V20 and V28 have few instances of fraud, so I removed those two variables, but it did not improve the accuracy
#remove V20 and V28
#X = X.drop(['V20', 'V28'], axis=1)

#After visualizing the data,
#remove rows that contain a value greater than 2.5 in the V24 column, because there are no frauds above 2.5 value
#And
#remove rows that contain a value greater than 20 in the Amount column, because there are no frauds above 20
#remove rows that contain a value greater than 5k in the Amount column, because there are no frauds above 5k

list_of_index = []
for index, row in X.iterrows():
    #if row['Amount'] > 20: #this filter gave me 94% accuracy with 29 columns, but that got rid of alot of fraud cases in the data set...
    if row['Amount'] > 3000 or row['V2'] < -25: 
        list_of_index.append(index)
# print(list_of_index)
# for index in list_of_index:
#     print(X['Amount'][index])
X = X.drop(list_of_index)
y = y.drop(list_of_index)


# In[63]:


print(X.shape)
print(y.shape)
num_of_frauds_after_filtering = 0
for value in y:
    if value == 1:
        num_of_frauds_after_filtering = num_of_frauds_after_filtering + 1
if num_of_frauds_before_filtering == num_of_frauds_after_filtering:
    print("I didn't remove any cases of fraud within the data, only removed cases of non fraud")
print(num_of_frauds_after_filtering)


# In[64]:


# Define the model
model = LogisticRegression()

# Define the splitter for splitting the data in a train set and a test set
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=0)

# Loop through the splits (only one)
for train_indices, test_indices in splitter.split(X, y):
    # Select the train and test data
    X_train, y_train = X.iloc[train_indices], y.iloc[train_indices]
    X_test, y_test = X.iloc[test_indices], y.iloc[test_indices]
    
    # Normalize the data
    X_train = normalize(X_train)
    X_test = normalize(X_test)
    
    # Fit and predict!
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # And finally: show the results
    print(classification_report(y_test, y_pred))


# In[ ]:




