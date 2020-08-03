#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")


# In[2]:


data = pd.read_csv('spambase/spambase.data', header=None)


# In[3]:


data = data.fillna(0)


# In[4]:


data


# # Part a

# In[6]:


print('There are a total of {} data points with {} features'.format(data.shape[0],data.shape[1]-1))


# In[5]:


print('There are {} spam and {} regular emails'.format(len(data[data[57]==1]),len(data[data[57]==0])))


# # Part b

# In[23]:


# Fit a CART model
X = data.iloc[:,:-1].values
Y = data.iloc[:,-1].values


clf = clf = tree.DecisionTreeClassifier(max_depth=5)
clf = clf.fit(X, Y)


# In[28]:


fig, ax = plt.subplots(figsize=(15, 15))
tree.plot_tree(clf,class_names = ['Spam','Non-Spam'], fontsize=12, filled=True) 
plt.show()


# # Part c

# In[31]:


xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.2)


# In[35]:


# Fit a CART and Random Forest model for multiple tree sizes
treesize = []
carty = []
rfy = []


for i in range(2,100):
    
    treesize.append(i)
    
    clf1 = tree.DecisionTreeClassifier(max_depth=i)
    clf1.fit(xtrain, ytrain)
    prediction = clf1.predict(xtest)
    cartauc = roc_auc_score(ytest, prediction)
    
    carty.append(cartauc)
    
    clf2 = RandomForestClassifier(max_depth=i)
    clf2.fit(xtrain, ytrain)
    prediction = clf2.predict(xtest)
    rfauc = roc_auc_score(ytest, prediction)
    
    rfy.append(rfauc)
    


# In[42]:


plt.plot(treesize, carty, rfy)
plt.xlabel('Tree Size')
plt.ylabel('AUC')
plt.legend(['CART','Random Forest'])


# In[ ]:




