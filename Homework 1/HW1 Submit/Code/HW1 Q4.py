#!/usr/bin/env python
# coding: utf-8

# In[39]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[74]:


# Read in data
df = pd.read_csv("data/food-consumption.csv")
df.head()


# In[75]:


# Clean the data
df1 = df.copy()
removeRows = ['Sweden','Finland','Spain']

df1 = df1[~df1['Country'].isin(removeRows)].reset_index(drop=True)
df1


# In[76]:


# Checking if there is any missing data
df1.isna().any()


# In[77]:


columnNames = df1.columns.drop('Country')


# In[78]:


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Separating out the features
x = df1.loc[:, columnNames].values
# Standardizing the features
x = StandardScaler().fit_transform(x)

pca = PCA(n_components=2)

# principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['Principal component 1', 'Principal component 2'])


# In[79]:


pca.fit(x)
print("The principal components are {}.".format(pca.components_))
print("The explained variance is {}".format(pca.explained_variance_))
principalComponents = pca.fit_transform(x)
variance = pca.explained_variance_ratio_ #calculate variance ratios
print("The explained variance ratio for each component is {}".format(variance))
var=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=3)*100)
var #cumulative sum of variance explained with [n] features
print("The cumulative variance for the 2 components is {:.2f}%".format(var[1]))


# In[80]:


df = pd.DataFrame(pca.components_,columns=df1.columns[1:],index = ['Principal Component 1','Principal Component 2'])
df= df.transpose()
df['Food Item'] = df.index

ax = df.plot.bar(x='Food Item', y='Principal Component 1', rot=0,figsize = (18,8))
plt.title("Principal Component 1")
plt.xticks(fontsize=12, rotation=45)
plt.show()

ax = df.plot.bar(x='Food Item', y='Principal Component 2', rot=0,figsize = (18,8))
plt.title("Principal Component 2")
plt.xticks(fontsize=12, rotation=45)
plt.show()


# In[81]:


components = pd.DataFrame(pca.components_.T)
components = components.rename ({0:"Principal Component 1",1:"Principal Component 2"}, axis = 'columns')
plt.figure(figsize=(16,10))
p1 = sns.scatterplot(
    x="Principal Component 1", y="Principal Component 2",
    palette=sns.color_palette("hls", 10),
    data=components,
    legend="full",
    alpha=0.7
)
foods = df1.columns.values
foods = np.delete(foods,0)

for i in range(0,len(foods)):
    p1.text(components['Principal Component 1'][i], 
            components['Principal Component 2'][i], 
            foods[i])


# In[82]:


import seaborn as sns
plt.figure(figsize=(16,10))
p1 = sns.scatterplot(
    x="Principal component 1", y="Principal component 2",
    palette=sns.color_palette("hls", 10),
    data=principalDf,
    legend="full",
    alpha=0.7
)

for i in range(0,len(df1)):
    p1.text(principalDf['Principal component 1'][i]+0.2, 
            principalDf['Principal component 2'][i], 
            df1['Country'][i])


# In[ ]:




