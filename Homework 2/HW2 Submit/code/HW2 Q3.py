#!/usr/bin/env python
# coding: utf-8

# # Q3

# ## Part a

# In[1]:


import pandas as pd
import scipy.io as sio
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import norm, multivariate_normal
import math
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# In[2]:


data = sio.loadmat('data/data.mat')


# In[3]:


images = data['data']


# In[4]:


testImage1 = images[:,0].reshape(28,28)
testImage2 = images[:,1900].reshape(28,28)


# In[5]:


fig, (ax1,ax2) = plt.subplots(1,2)
ax1.imshow(testImage1.T,cmap='gray')
ax2.imshow(testImage2.T,cmap='gray')


# # EM Algorithm

# In[6]:


X = images[:,0].reshape(784,1)
X.shape


# In[7]:


X = images


# In[8]:


nrow, ncol = X.shape  
r = 100
piVec = np.ones(2)/2 


# In[9]:


# Initializing covariances matrices
cov1 = np.identity(nrow)
cov2 = np.identity(nrow)


# In[10]:


# Initializing mean vectors
muVec1 = np.random.normal(0, 1, nrow)
muVec2 = np.random.normal(0, 1, nrow)


# In[11]:


# Initializing vectors to store log-likelihoods
logLikelihood = 0
likelihoodVec = []
hasConverged = False
counter = 0


# In[12]:


# Looping through the EM algorithm until convergence
while ~hasConverged:
#   Finding eigenvalues
    eigVal1, eigVec1 = np.linalg.eig(cov1)
    eigVal2, eigVec2 = np.linalg.eig(cov2)
    
    eigVal1 = eigVal1.real
    eigVec1 = eigVec1.real
    eigVal2 = eigVal2.real
    eigVec2 = eigVec2.real
    
#   Using low-rank approximation
    indexSort1 = eigVal1.argsort()[::-1]
    eigVal1 = eigVal1[indexSort1[0:r]]
    eigVec1 = eigVec1[:,indexSort1[0:r]]
    
    indexSort2 = eigVal2.argsort()[::-1]
    eigVal2 = eigVal2[indexSort2[0:r]]
    eigVec2 = eigVec2[:,indexSort2[0:r]]
    
    x1 = eigVec1.T @ X
    mu1 = eigVec1.T @ muVec1
    
    x2 = eigVec2.T @ X
    mu2 = eigVec2.T @ muVec2
    
#   Updating the m vector
    m1Vec = []
    m2Vec = []
    for i in range(r):
        m1Val = (x1[i]-mu1[i])**2/eigVal1[i]
        m2Val = (x2[i]-mu2[i])**2/eigVal2[i]
        m1Vec.append(m1Val)
        m2Vec.append(m2Val)
    
    m1 = np.sum(np.stack(m1Vec),axis=0)
    m2 = np.sum(np.stack(m2Vec),axis=0)
    
#   Updating the D vector
    D1 = 1
    D2 = 1
    for i in range(r):
        D1 = D1 * eigVal1[i]**(-0.5) 
        D2 = D2 * eigVal2[i]**(-0.5) 
        
        
#   Updating the tau vector
    t1 = []
    t2 = []
    for i in range(ncol):
        t1.append(piVec[0] * D1 * math.exp(-0.5 * m1[i]))
        t2.append(piVec[1] * D2 * math.exp(-0.5 * m2[i]))
    t1 = np.array(t1)
    t2 = np.array(t2)
    
#   Normalizing
    C = []
    for i in range(len(t1)):
        C.append(t1[i]+t2[i])
    t1Normalized = t1/np.array(C)
    t2Normalized = t2/np.array(C)
    
    currLogLikelihood = np.log(np.sum(C))
    
    if abs(currLogLikelihood - logLikelihood) < 1e-4:
        hasConverged = True
    else:
#       Maximization step
        logLikelihood = currLogLikelihood
        
        piVec[0] = np.sum(t1Normalized)/ncol
        piVec[1] = np.sum(t2Normalized)/ncol
        
        muVec1 = np.sum(t1Normalized*X,axis=1)/np.sum(t1Normalized)
        muVec2 = np.sum(t2Normalized*X,axis=1)/np.sum(t2Normalized)
        
        temp1 = (X-np.tile(muVec1, (ncol, 1)).T)
        cov1 = ((t1Normalized*temp1) @temp1.T)/np.sum(t1Normalized)
        
        temp2 = (X-np.tile(muVec2, (ncol, 1)).T)
        cov2 = ((t2Normalized*temp2) @temp2.T)/np.sum(t2Normalized)
    
        
    
    likelihoodVec.append(currLogLikelihood)
    print(counter, currLogLikelihood)
    counter += 1
#   Stop after 50 iterations
    if counter == 50:
        break
    
    


# ## Part b

# In[384]:


plt.plot(np.linspace(0,20),likelihoodVec,'o-')
plt.xlabel("Iteration")
plt.ylabel("Log Likelihood")


# ## Part c

# In[13]:


fig, (ax1,ax2) = plt.subplots(1,2)
ax1.imshow(muVec1.reshape(28,28).T,cmap='gray')

ax2.imshow(muVec2.reshape(28,28).T,cmap='gray')
ax1.grid(False)
ax2.grid(False)


# In[14]:


print("The weight for the images of 6 and 2 are {:.2f} and {:.2f}, respectively ".format(piVec[0],piVec[1]))


# ## Part d

# In[15]:


labelData = sio.loadmat('data/label.mat')


# In[16]:


labels = labelData['trueLabel'][0]


# In[17]:


predictions = []

for i in range(ncol):
    if t1[i] > t2[i]:
        predictions.append(6)
    else:
        predictions.append(2)


# In[19]:


GMMclassification = confusion_matrix(labels,predictions)
misclassified = (GMMclassification[0][1]+GMMclassification[1][0])/np.sum(GMMclassification)*100


# In[20]:


misclassified


# In[21]:


kmeansModel = KMeans(n_clusters=2,random_state=1)


# In[22]:


kmeansModel.fit(X.T)


# In[23]:


kmeansLabels = kmeansModel.labels_


# In[24]:


kmeanPredictions = []


# In[25]:


for i in kmeansLabels:
    if i == 0:
        kmeanPredictions.append(2)
    else:
        kmeanPredictions.append(6)


# In[26]:


kmeansclassification = confusion_matrix(labels,kmeanPredictions)
misclassifiedkmeans= (kmeansclassification[0][1]+kmeansclassification[1][0])/np.sum(kmeansclassification)*100


# In[28]:


misclassifiedkmeans


# ## Part e

# In[29]:


pca = PCA(5)


# In[30]:


XPCA = pca.fit_transform(X.T).T


# In[31]:


X = XPCA


# In[32]:


# Re-running the EM algorithm on the new PCA dataset
nrow, ncol = X.shape  
r = 5
piVec = np.ones(2)/2 

cov1 = np.identity(nrow)
cov2 = np.identity(nrow)
muVec1 = np.random.normal(0, 1, nrow)
muVec2 = np.random.normal(0, 1, nrow)

logLikelihood = 0
likelihoodVec = []
hasConverged = False
counter = 0

while ~hasConverged:
    
    eigVal1, eigVec1 = np.linalg.eig(cov1)
    eigVal2, eigVec2 = np.linalg.eig(cov2)
    
    eigVal1 = eigVal1.real
    eigVec1 = eigVec1.real
    eigVal2 = eigVal2.real
    eigVec2 = eigVec2.real
    
    indexSort1 = eigVal1.argsort()[::-1]
    eigVal1 = eigVal1[indexSort1[0:r]]
    eigVec1 = eigVec1[:,indexSort1[0:r]]
    
    indexSort2 = eigVal2.argsort()[::-1]
    eigVal2 = eigVal2[indexSort2[0:r]]
    eigVec2 = eigVec2[:,indexSort2[0:r]]
    
    x1 = eigVec1.T @ X
    mu1 = eigVec1.T @ muVec1
    
    x2 = eigVec2.T @ X
    mu2 = eigVec2.T @ muVec2
    
    
    m1Vec = []
    m2Vec = []
    for i in range(r):
        m1Val = (x1[i]-mu1[i])**2/eigVal1[i]
        m2Val = (x2[i]-mu2[i])**2/eigVal2[i]
        
        m1Vec.append(m1Val)
        m2Vec.append(m2Val)
    
    m1 = np.sum(np.stack(m1Vec),axis=0)
    m2 = np.sum(np.stack(m2Vec),axis=0)
    
    
    D1 = 1
    D2 = 1
    for i in range(r):
        D1 = D1 * eigVal1[i]**(-0.5) 
        D2 = D2 * eigVal2[i]**(-0.5) 
        
    t1 = []
    t2 = []
    
    for i in range(ncol):
        t1.append(piVec[0] * D1 * math.exp(-0.5 * m1[i]))
        t2.append(piVec[1] * D2 * math.exp(-0.5 * m2[i]))
        
    t1 = np.array(t1)
    t2 = np.array(t2)
    
    
    C = []
    for i in range(len(t1)):
        C.append(t1[i]+t2[i])

    t1Normalized = t1/np.array(C)
    t2Normalized = t2/np.array(C)
    
    currLogLikelihood = np.log(np.sum(C))
    
    if abs(currLogLikelihood - logLikelihood) < 1e-4:
        hasConverged = True
    else:
        logLikelihood = currLogLikelihood
        
        piVec[0] = np.sum(t1Normalized)/ncol
        piVec[1] = np.sum(t2Normalized)/ncol
        
        muVec1 = np.sum(t1Normalized*X,axis=1)/np.sum(t1Normalized)
        muVec2 = np.sum(t2Normalized*X,axis=1)/np.sum(t2Normalized)
        
        temp1 = (X-np.tile(muVec1, (ncol, 1)).T)
        cov1 = ((t1Normalized*temp1) @temp1.T)/np.sum(t1Normalized)
        
        temp2 = (X-np.tile(muVec2, (ncol, 1)).T)
        cov2 = ((t2Normalized*temp2) @temp2.T)/np.sum(t2Normalized)
    
        
    
    likelihoodVec.append(currLogLikelihood)
    print(counter, currLogLikelihood)
    counter += 1
    if counter == 50:
        break


# In[33]:


plt.plot(np.linspace(0,20),likelihoodVec,'o-')
plt.xlabel("Iteration")
plt.ylabel("Log Likelihood")


# In[36]:


predictions = []

for i in range(ncol):
    if t1[i] > t2[i]:
        predictions.append(2)
    else:
        predictions.append(6)

pcaclassification = confusion_matrix(labels,predictions)
misclassifiedpca = (pcaclassification[0][1]+pcaclassification[1][0])/np.sum(pcaclassification)*100


# In[37]:


misclassifiedpca

