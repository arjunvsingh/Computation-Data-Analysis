#!/usr/bin/env python
# coding: utf-8

# In[113]:


import numpy as np
import time


# In[114]:


def init_medoids(X, k):
#   Randomly select k points from the image as the starting centers
    centers = np.random.choice(len(X),k,replace=False)
#     print(np.repeat(centers,k))
#     temp = np.repeat(centers,k)
#     return X[temp,:]
    return X[centers,:]


# In[115]:


# Compute  distance. I've included the metric p in the input to allow various norms to compute the 
# distance (rather than just Euclidean)
def compute_distance(X, centers, p):
    m = len(X)
    
    center_shape = centers.shape
    if len(center_shape)==1:
        centers = centers.reshape((1,len(centers)))
    
    s=len(centers)
    
    S = np.empty((m,s))
    for i in range(m):
        S[i,:] = np.linalg.norm(X[i,:]-centers,ord=p,axis=1)**p
    return S


# In[116]:


# Another approach to calculating distance between points in cluster. Avoids the loop in the previous distance 
# calculation approach.
def distance_calc (cluster_points,center,p,k):
    dist = []
    label = []
    min_dist = []
    
    main_array = list(range(k))
    pixel_temp = np.tile(center,(len(cluster_points),1))
    
    main_array= np.linalg.norm(cluster_points - pixel_temp, ord=p,axis=1)
    min_dist = main_array

    return min_dist


# In[117]:


def assign_cluster_labels(S):
    return np.argmin(S,axis=1)


# In[118]:


# Updates the cluster centers by finding the point with the least distance to all other points.
def update_centers(X, centers, p,k):
    
    S = compute_distance(X, centers,p)
    labels = assign_cluster_labels(S)
    
    curr_centers = centers
#   Iterate over all clusters
    for i in set(labels):
        cluster_points = X[labels==i]
        cluster_points = np.unique(cluster_points,axis=0)
        avg_distance = np.sum(distance_calc(cluster_points, centers[i],p,k))
#       Iterate over all points in each cluster
        for points in cluster_points:
            new_distance = np.sum(distance_calc(cluster_points,points,p,k))
#           If distance from current point to all other points in the cluster is lower than previous minimum
#           then update cluster center to the new point 

            if new_distance < avg_distance:
                avg_distance = new_distance
                curr_centers[i] = points
    
    return curr_centers


# In[119]:


# Calculate the within clusters sum of squares
def WCSS(S):
    return np.sum(np.amin(S,axis=1))


# In[121]:


def kmedoids(X, k,p,starting_centers=None,max_steps=np.inf):
    start = time.time()
#   Begin by initializing starting points
    if starting_centers is None:
        centers = init_medoids(X, k)
    else:
        centers = starting_centers
        
    converged = False
    labels = np.zeros(len(X))
    i = 1
    wc= 99999999999
#   Begin loop for algorithm
    while (not converged) and (i <= max_steps):
        oldwcss = wc
        old_centers = centers
        
        S = compute_distance(X, old_centers,p)
        
        labels = assign_cluster_labels(S)
        
        centers = update_centers(X, centers,p,k)
        
        wc = WCSS(S)
#       If current step's WCSS is less than a 5% imporvement over previous step, terminate
        if wc > 0.95*oldwcss:
            converged = True
        else:
            converged = False
        print ("iteration", i, "WCSS = ", WCSS (S))
        i += 1
                  
    stop = time.time()
    print("Time taken = {:.2f} seconds".format(stop-start))
    return labels


# In[62]:


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


# In[63]:


from PIL import Image
def read_img(path):
    """
    Read image and store it as an array, given the image path. 
    Returns the 3 dimensional image array.
    """
    img = Image.open(path)
    img_arr = np.array(img, dtype='int32')
#     img_arr = np.array(img.getdata())

    img.close()
    return img_arr


# In[64]:


def runKMedoidsAlgorithm(pixels,k):
    print("Running k-medoids algorithm...")
#   Read in image
    image = read_img(pixels)
    r, c, l = image.shape
#   Flatten image
    img_reshaped = np.reshape(image, (r*c, l), order="C")
#   Choose norm criteria
    p=2
#   If the value k is too high, reduce it
    if k>= (len(img_reshaped)/3):
        k = (len(img_reshaped)/3)
        
#   Run algorithm
    labels = kmedoids(img_reshaped, k,p, starting_centers=None)
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


# In[104]:


lab, cen = runKMedoidsAlgorithm("data/beach.bmp",2)


# In[105]:


lab, cen = runKMedoidsAlgorithm("data/beach.bmp",16)


# In[106]:


lab, cen = runKMedoidsAlgorithm("data/beach.bmp",32)


# In[107]:


lab, cen = runKMedoidsAlgorithm("data/football.bmp",4)


# In[108]:


lab, cen = runKMedoidsAlgorithm("data/football.bmp",14)


# In[109]:


lab, cen = runKMedoidsAlgorithm("data/football.bmp",50)

