#!/usr/bin/env python
# coding: utf-8

# # Q2

# ## Part a

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.neighbors import KernelDensity


# In[2]:


data = pd.read_csv('data/n90pol.csv')


# In[3]:


data.head()


# In[4]:


x = data['amygdala']
y = data['acc']
weight = data['orientation']


# In[5]:


plt.figure(figsize=(8,8))
plt.hist2d(x,y,weights=weight,bins=20)
plt.xlabel('Amygdala')
plt.ylabel('ACC')
plt.show()


# In[6]:


g = (sns.jointplot(x, y, kind="hex").set_axis_labels("Amygdala", "ACC"))


# ## Part b

# In[7]:


# instantiate and fit the KDE model
kde = KernelDensity(bandwidth=1.0, kernel='gaussian')
kde.fit(data)


# In[8]:


# Using Seaborn's kdeplot
ax = sns.kdeplot(x, y,cmap="Blues", kernel='gau',shade=True, shade_lowest=False,bw=0.01)


# ## Part c

# In[9]:


# Finding the unique values for the orientation
data['orientation'].unique()


# In[10]:


o2 = data[data['orientation']==2]
o3 = data[data['orientation']==3]
o4 = data[data['orientation']==4]
o5 = data[data['orientation']==5]


# In[11]:


fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True,figsize=(20,10))

sns.distplot(o2['amygdala'],hist=False,bins=5, label='Orientation 2',ax=ax1)
sns.distplot(o3['amygdala'],hist=False,bins=5, label='Orientation 3',ax=ax1)
sns.distplot(o4['amygdala'],hist=False,bins=5, label='Orientation 4',ax=ax1)
sns.distplot(o5['amygdala'],hist=False,bins=5, label='Orientation 5',ax=ax1)

sns.distplot(o2['acc'],hist=False,bins=5, label='Orientation 2',ax=ax2)
sns.distplot(o3['acc'],hist=False,bins=5, label='Orientation 3',ax=ax2)
sns.distplot(o4['acc'],hist=False,bins=5, label='Orientation 4',ax=ax2)
sns.distplot(o5['acc'],hist=False,bins=5, label='Orientation 5',ax=ax2)


ax1.legend(prop={'size': 12})
ax1.set_title('Amygdala Density vs Orientation')
ax1.set_xlabel('Amygdala')
ax1.set_ylabel('Density')

ax2.legend(prop={'size': 12})
ax2.set_title('ACC Density vs Orientation')
ax2.set_xlabel('ACC')
ax2.set_ylabel('Density')

