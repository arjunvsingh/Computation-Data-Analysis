#!/usr/bin/env python
# coding: utf-8

# # Q2
# 
# ## Part a

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import scipy.io as sio


# In[2]:


data = pd.read_csv("marriage.csv",header=None)


# In[3]:


x = data.iloc[:,:-1]
y = data.iloc[:,-1]


# In[4]:


xtrain, xtest, ytrain,ytest = train_test_split(x,y, train_size = 0.8)


# In[5]:


NBClassifier = GaussianNB()
LRClassifier = LogisticRegression()
KNNClassifier = KNeighborsClassifier()


# In[6]:


NBClassifier.fit(xtrain,ytrain)
LRClassifier.fit(xtrain,ytrain)
KNNClassifier.fit(xtrain,ytrain)


# In[7]:


print("The accuracy of the Naive Bayes classifier is {:.2f}%".format(NBClassifier.score(xtest,ytest)*100))
print("The accuracy of the Logistic Regression classifier is {:.2f}%".format(LRClassifier.score(xtest,ytest)*100))
print("The accuracy of the K Nearest Neighbours classifier is {:.2f}%".format(KNNClassifier.score(xtest,ytest)*100))


# ## Part b
# Using first 2 features

# In[8]:


x = data.iloc[:,0:2]
y = data.iloc[:,-1]
xtrain, xtest, ytrain,ytest = train_test_split(x,y, train_size = 0.8)


# In[9]:


NBClassifier = GaussianNB()
LRClassifier = LogisticRegression()
KNNClassifier = KNeighborsClassifier()
NBClassifier.fit(xtrain,ytrain)
LRClassifier.fit(xtrain,ytrain)
KNNClassifier.fit(xtrain,ytrain)


# In[10]:


print("The accuracy of the Naive Bayes classifier is {:.2f}%".format(NBClassifier.score(xtest,ytest)*100))
print("The accuracy of the Logistic Regression classifier is {:.2f}%".format(LRClassifier.score(xtest,ytest)*100))
print("The accuracy of the K Nearest Neighbours classifier is {:.2f}%".format(KNNClassifier.score(xtest,ytest)*100))


# In[11]:


xtrain1 = xtrain.values
xtest1 = xtest.values
ytrain1 = ytrain.values
ytest1 = ytest.values
X1 = xtest.values
h=0.1


# In[12]:


# Code referenced from sklearn:
# https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
x_min, x_max = X1[:, 0].min() - .5, X1[:, 0].max() + .5
y_min, y_max = X1[:, 1].min() - .5, X1[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# just plot the dataset first
cm = plt.cm.RdBu
cm_bright = ListedColormap(['#FF0000', '#0000FF'])
ax = plt.subplot(1,  1, 1)

ax.set_title("Naive Bayes Classifier")


Z = NBClassifier.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

# Put the result into a color plot
Z = Z.reshape(xx.shape)
ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

# Plot the training points
# ax.scatter(xtrain1[:, 0], xtrain1[:, 1], c=ytrain1, cmap=cm_bright,
#            edgecolors='k')
# Plot the testing points
ax.scatter(xtest1[:, 0], xtest1[:, 1], c=ytest1, cmap=cm_bright,
           edgecolors='k', alpha=0.6)

ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_xticks(())
ax.set_yticks(())


# In[13]:


x_min, x_max = X1[:, 0].min() - .5, X1[:, 0].max() + .5
y_min, y_max = X1[:, 1].min() - .5, X1[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# just plot the dataset first
cm = plt.cm.RdBu
cm_bright = ListedColormap(['#FF0000', '#0000FF'])
ax = plt.subplot(1,  1, 1)

ax.set_title("Logistic Regression Classifier")


Z = LRClassifier.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

# Put the result into a color plot
Z = Z.reshape(xx.shape)
ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

# Plot the training points
# ax.scatter(xtrain1[:, 0], xtrain1[:, 1], c=ytrain1, cmap=cm_bright,
#            edgecolors='k')
# Plot the testing points
ax.scatter(xtest1[:, 0], xtest1[:, 1], c=ytest1, cmap=cm_bright,
           edgecolors='k', alpha=0.6)

ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_xticks(())
ax.set_yticks(())


# In[14]:


h=0.2
x_min, x_max = X1[:, 0].min() - .5, X1[:, 0].max() + .5
y_min, y_max = X1[:, 1].min() - .5, X1[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# just plot the dataset first
cm = plt.cm.RdBu
cm_bright = ListedColormap(['#FF0000', '#0000FF'])
ax = plt.subplot(1,  1, 1)

ax.set_title("KNN Classifier")


Z = KNNClassifier.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

# Put the result into a color plot
Z = Z.reshape(xx.shape)
ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

# Plot the training points
# ax.scatter(xtrain1[:, 0], xtrain1[:, 1], c=ytrain1, cmap=cm_bright,
#            edgecolors='k')
# Plot the testing points
ax.scatter(xtest1[:, 0], xtest1[:, 1], c=ytest1, cmap=cm_bright,
           edgecolors='k', alpha=0.6)

ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_xticks(())
ax.set_yticks(())


# # Part 2 - MNIST

# In[15]:


data = sio.loadmat('data/data.mat')
label = sio.loadmat('data/label.mat')


# In[16]:


images = data['data']
labels = label['trueLabel'][0]


# In[17]:


labels[labels == 2] = 0
labels[labels == 6] = 1


# In[18]:


xtrain, xtest, ytrain,ytest = train_test_split(images.T, labels, train_size = 0.8, random_state = 10)


# In[19]:


NBClassifier = GaussianNB()
LRClassifier = LogisticRegression()
KNNClassifier = KNeighborsClassifier()
NBClassifier.fit(xtrain,ytrain)
LRClassifier.fit(xtrain,ytrain)
KNNClassifier.fit(xtrain,ytrain)


# In[20]:


print("The accuracy of the Naive Bayes classifier is {:.2f}%".format(NBClassifier.score(xtest,ytest)*100))
print("The accuracy of the Logistic Regression classifier is {:.2f}%".format(LRClassifier.score(xtest,ytest)*100))
print("The accuracy of the K Nearest Neighbours classifier is {:.2f}%".format(KNNClassifier.score(xtest,ytest)*100))


# In[ ]:




