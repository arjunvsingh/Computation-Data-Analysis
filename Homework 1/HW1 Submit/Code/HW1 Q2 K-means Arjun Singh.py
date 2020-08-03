#!/usr/bin/env python
# coding: utf-8

# K-Means algorithm. Please note that I have reused some of my old code from CSE 6040.

# In[1]:


import numpy as np
import time


# In[32]:


#   Randomly selects k points from the input image
def init_centers(X, k):

    centers = np.random.choice(len(X),k,replace=False)
#     centers = np.random.choice(len(X),1,replace=False)
#     return X[np.repeat(centers,k),:]
    return X[centers,:]


# In[55]:


# Compute Euclidean distance
def compute_d2(X, centers):

    m = len(X)
    s=len(centers)
    
    S = np.empty((m,s))
    for i in range(m):
        S[i,:] = np.linalg.norm(X[i,:]-centers,ord=2,axis=1)**2
    
    return S


# In[56]:


def assign_cluster_labels(S):
    return np.argmin(S,axis=1)


# In[57]:


def update_centers(X, y):
    # X[:m, :d] == m points, each of dimension d
    # y[:m] == cluster labels
    m, d = X.shape
    k = max(y) + 1
    assert m == len(y)
    assert (min(y) >= 0)
    
    centers = np.empty((k, d))
    for j in range(k):
        # Compute the new center of cluster j,
        # i.e., centers[j, :d].
        centers[j,:] = np.mean(X[j==y],axis=0)
    return centers


# In[58]:


# Calculate the within clusters sum of squares
def WCSS(S):
    return np.sum(np.amin(S,axis=1))


# In[23]:


# The algorithm has converged if there is no change in the cluster centers in the next iteration
def has_converged(old_centers, centers):
    return set([tuple(x) for x in old_centers]) == set([tuple(x) for x in centers])


# In[24]:


def kmeans(X, k,starting_centers=None,max_steps=np.inf):
    start = time.time()
    
    if starting_centers is None:
        centers = init_centers(X, k)
    else:
        centers = starting_centers
        
    converged = False
    labels = np.zeros(len(X))
    i = 1
    while (not converged) and (i <= max_steps):
        old_centers = centers
        S = compute_d2(X, old_centers)
        labels = assign_cluster_labels(S)
        centers = update_centers(X, labels)
        wc = WCSS(S)
        
        converged = has_converged(old_centers,centers)
        print ("iteration", i, "WCSS = ", WCSS (S))
        i += 1
    stop = time.time()
    print("Time taken = {:.2f} seconds".format(stop-start))
    return labels


# In[25]:


from matplotlib.pyplot import imshow
get_ipython().run_line_magic('matplotlib', 'inline')
def display_image(arr):
    """
    display the image
    input : 3 dimensional array
    """
    arr = arr.astype(dtype='uint8')
    img = Image.fromarray(arr, 'RGB')
    imshow(np.asarray(img))


# In[26]:


from PIL import Image
def read_img(path):
    """
    Read image and store it as an array, given the image path. 
    Returns the 3 dimensional image array.
    """
    img = Image.open(path)
    img_arr = np.array(img, dtype='int32')
    img.close()
    return img_arr


# In[27]:


def runKMeansAlgorithm(pixels,k):
    print("Running k-means algorithm...")
#   Read image
    image = read_img(pixels)
    r, c, l = image.shape
#   Flatten image
    img_reshaped = np.reshape(image, (r*c, l), order="C")
#   Run k-means algorithm
    labels = kmeans(img_reshaped, k, starting_centers=None)
    ind = np.column_stack((img_reshaped, labels))
    centers = {}
    for i in set(labels):
        c = ind[ind[:,3] == i].mean(axis=0)
        centers[i] = c[:3]
        
    img_clustered = np.array([centers[i] for i in labels])
    r, c, l = image.shape
    img_disp = np.reshape(img_clustered, (r, c, l), order="C")
    
    print('Image with k = ' + str(k) + " clusters...")
    display_image(img_disp)
    print("The labels are: {}".format(labels.T))
    print("The cluster centers are {}".format(centers))
    return labels,centers


# In[36]:


lab, cen = runKMeansAlgorithm("data/football.bmp",3)


# In[37]:


lab, cen = runKMeansAlgorithm("data/football.bmp",16)


# In[38]:


lab, cen = runKMeansAlgorithm("data/football.bmp",32)


# In[39]:


lab, cen = runKMeansAlgorithm("data/TechTower.jpg",2)


# In[40]:


lab, cen = runKMeansAlgorithm("data/TechTower.jpg",12)


# In[54]:


lab, cen = runKMeansAlgorithm("data/beach.bmp",32)


# In[ ]:




