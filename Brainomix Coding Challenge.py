#!/usr/bin/env python
# coding: utf-8

# ## BRAINOMIX CODING CHALLENGE
# #### Shruti Shikhare
# ### OUTLINE
# 
# 1. Explore the CT images, get header, get pixel and voxel coordinates from slices
# 2. Intensity thresholding to locate lungs
# 3. Find the contours of the lung edge
# 4. Find the lung area from these contours and create a mask
# 5. Segment vessels to calculate the vessel:lung area ratio
# 6. k-means clustering to classify the data points
# 

# In[623]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import sys
import os
import glob
import random
import time
# import PIL as Image 
import nibabel as nib
from skimage.io import imread
import skimage.io
import skimage.measure
from scipy.spatial import ConvexHull
from skimage.draw import polygon


# In[624]:


# DATA VISUALISATION
data_path = "/Users/shrinivasshikhare/Desktop/brainomix/Images"
data_map=[]
for sub_dir_path in glob.glob(data_path+"*"):
    if os.path.isdir(sub_dir_path):
        filename = os.path.join(data_path)

for sub_dir_path in glob.glob(data_path):
    if os.path.isdir(sub_dir_path):
        dirname = sub_dir_path.split("/")[-1]
        for filename in os.listdir(sub_dir_path):
            image_path = sub_dir_path + "/" + filename
            data_map.extend([filename,image_path])

df = pd.DataFrame({"filename":data_map[::2], "path":data_map[1::2]})

df['lung'] = df['filename'].apply(lambda s: s[0:-7].strip()).astype(str)

df = df.drop(labels=1, axis=0)
df=df.reset_index(drop=True)
df.head()

# the dataframe now has the path of all nine images including the filename and lung number


# In[625]:


# explore teh images, plot the last slice of all the images in the dataset

def plot_all_images(df):
    for i in range(8):
        test_image = nib.load(df['path'][i])
        test_image_data=test_image.get_fdata()
        i,j,k = test_image_data.shape
#         sx,sy,sz = test_image.header.get_zooms()
#         volume = sx*sy *sz        # volume of one voxel
#         print("volume of voxel", df['path'][i],": ", volume)
        
        plt.figure()
        plt.imshow(test_image_data[:,:,k-1].T,cmap='gray')
        plt.title('Raw Image')
        
plot_all_images(df)


# In[626]:


## this function shows the image in a window of intensity (units HU)
mean_lung_vol=[]
mean_vessel_vol=[]
mean_vol_ratio=[]

for i in range(9):
    lung_volumes=[]
    vessel_volumes=[]
    ratios=[]
    
    test_image = nib.load(df['path'][i])
    test_image_data = test_image.get_fdata()
    ii,jj,k = test_image_data.shape
    
    for slice_num in range(k):
        
        ## to find the contours and individualise the two lungs
        
        contours, test_image_range = intensity_thresholding(test_image_data[:,:,slice_num].T, min=-1000, max=-300)
        sorted_contours = lung_contours(contours)
        
        ## now that we have identified the lungs, we will now create a binary mask and save it 

        lung_mask = np.zeros(test_image_data.shape)
        for contour in sorted_contours:
            rr, cc = polygon(contour[:, 0], contour[:, 1], lung_mask.shape)
            lung_mask[rr, cc] = 1

        lung_mask = (lung_mask.T).astype('int')
        lung_mask[lung_mask == 1] = 255
        lung_mask = np.transpose(lung_mask)

#         plt.figure()
#         plt.imshow(lung_mask[:,:,slice_num],cmap='gray')
#         plt.title('Mask of Lung')
#         plt.show()

        lung_mask_nifti = nib.Nifti1Image(lung_mask, test_image.affine)
        nib.save(lung_mask_nifti, filename='Images/masks/lung' + str(i) + 'slice' + str(slice_num) + '_mask.nii.gz')

        ## to compute the area of the lungs, vessels, and their volume ratio
        
        lung_mask[lung_mask>=1]=1
        lung_mask[lung_mask<1]=0
        lung_pixels = np.sum(lung_mask)
        pixdim = find_voxels(test_image)
        lung_volume = (lung_pixels*pixdim[0]*pixdim[1]*pixdim[2])/1000
        lung_volumes.append(lung_volume)

        vessel_volume, ratio = vessels_and_ratio(lung_mask,test_image_data,lung_volume)
        vessel_volumes.append(vessel_volume)
        ratios.append(ratio)
        
#     plots(test_image_data,sorted_contours,vessels)
    mean_lung_vol.append(np.mean(np.array(lung_volumes)[:].astype(float))) #mean volumes over all slices per image
    mean_vessel_vol.append(np.mean(np.array(vessel_volumes)[:].astype(float)))
    mean_vol_ratio.append(np.mean(np.array(ratios)[:].astype(float)))
#     print(mean_lung_vol)
#     print(mean_vessel_vol)
#     print(mean_vol_ratio)


# In[580]:


## save lung volumes, vessel volumes in a dataframe

results = pd.DataFrame()
results['Lung']=df['lung'][0:9]
results['Lung Volumes']=mean_lung_vol
results['Vessel Volumes']=mean_vessel_vol
results['Vessel-Lung Ratio']=mean_vol_ratio
results.to_csv('results.csv')
results.head(10)


# In[622]:


## plot the lung and vessel mask, along with the 
for i in range(9):
    test_image = nib.load(df['path'][i])
    test_image_data = test_image.get_fdata()
    ii,jj,k = test_image_data.shape
    
    contours, test_image_range = intensity_thresholding(test_image_data[:,:,k-1], min=-1000, max=-300)
    sorted_contours = lung_contours(contours)
        
    ## now that we have identified the lungs, we will now create a binary mask and save it 

    lung_mask = np.zeros(test_image_data.shape)
    for contour in sorted_contours:
        rr, cc = polygon(contour[:, 0], contour[:, 1], lung_mask.shape)
        lung_mask[rr, cc] = 1

    lung_mask = (lung_mask.T).astype('int')
    lung_mask[lung_mask == 1] = 255
    lung_mask = np.transpose(lung_mask)

    ## to compute the area of the lungs, vessels, and their volume ratio

    lung_mask[lung_mask>=1]=1
    lung_mask[lung_mask<1]=0

    vessels = lung_mask * test_image_data
    vessels[vessels == 0] = -1000
    vessels[vessels >= -500] = 1
    vessels[vessels < -500] = 0
#     vessels.shape

    ## calulcate the ratio
    vessels[vessels>=1]=1

    fig,ax = plt.subplots(1,2,figsize=(10,10))

    ax[0].imshow(test_image_data[:,:,k-1].T, cmap='gray')
    ax[0].set_title('Lung Contours')
    for contour in sorted_contours:
        ax[0].plot(contour[:, 0], contour[:, 1])
    
    ax[1].imshow(lung_mask[:,:,k-1].T,cmap='gray')
    ax[1].set_title('Lung Image Mask')
    plt.savefig('lung mask'+str(i))
    plt.show()
    
    fig,ax = plt.subplots(1,2,figsize=(10,10))
    
    ax[0].imshow(vessels[:,:,k-1].T,cmap='gray')
    ax[0].set_title('Mask of Abnormal Lung Parenchyma')

    ax[1].imshow(test_image_data[:,:,k-1].T, cmap='gray')
    ax[1].imshow(vessels[:,:,k-1].T,alpha=0.6)
    ax[1].set_title('Superimposed View of Abnormal Lung Parenchyma')
    plt.savefig('vessel mask'+str(i))
    plt.show()


# In[590]:


# def show_slice_window(slice, level, window):
#     max = level + window/2
#     min = level - window/2
#     slice = slice.clip(min,max)
#     plt.figure()
#     plt.imshow(slice.T, cmap="gray", origin="lower")
# #     plt.savefig('L'+str(level)+'W'+str(window))

# show_slice_window(test_image_data[:,:,k-1], 0, 500)

## thresholding of intensity
# creating a histogram of image intensity

def intensity_plot(test_image_data):
    histogram, bin_edges = np.histogram(test_image_data, bins=300)

    plt.figure()
    plt.title("Grayscale Histogram")
    plt.xlabel("grayscale value")
    plt.ylabel("pixels")

    plt.plot(bin_edges[0:-1], histogram)
    plt.show()

## binarise image 

def intensity_thresholding(test_image_data, min=-1000, max=-300):
    
    test_image_range = test_image_data.clip(min, max)
    test_image_range[test_image_range != max] = 1
    test_image_range[test_image_range == max] = 0
    
#     plt.figure()
#     plt.imshow(test_image_range, cmap='gray',origin="lower")
#     plt.show()
#     intensity_plot(test_image_data)
    
    contours = skimage.measure.find_contours(test_image_range, 0.95)
    
    return contours, test_image_range

def lung_contours(contours):
    
    hull_vol=[]
    distances=[]
    all_contours=[]

    for contour in contours:
        ## calculate the distance between two adjacent contours
        dx = contour[0,1] - contour[-1,1]
        dy = contour[0,0] - contour[-1,0]
        distance = np.sqrt(np.power(dx,2) + np.power(dy,2))

        hull = ConvexHull(contour)  # convexhull returns the spatial data points of the contours

        if hull.volume > 2000 and distance < 1:
                all_contours.append(contour)
                hull_vol.append(hull.volume)

    sorted_contours=[]
    hull_vol, all_contours = (list(t) for t in zip(*sorted(zip(hull_vol, all_contours))))
    sorted_contours = sorted(all_contours, key=lambda contour: max(contour[0]))
    sorted_contours = sorted_contours[0:2]
    hull_vol.sort()


    return sorted_contours



def find_voxels(ct_img):    

    sx, sy, sz  = ct_img.header.get_zooms()
    
    return [sx,sy,sz]

### vessels
def vessels_and_ratio(lung_mask,test_image_data,lung_volume):
    vessels = lung_mask * test_image_data
    vessels[vessels == 0] = -1000
    vessels[vessels >= -500] = 1
    vessels[vessels < -500] = 0
#     vessels.shape

    ## calulcate the ratio
    vessels[vessels>=1]=1
    vessel_pixels = np.sum(vessels)
    pixdim = find_voxels(test_image)
    vessel_volume = (vessel_pixels * pixdim[0]*pixdim[1]*pixdim[2])/1000 #conversion to ml

    ratio = (vessel_volume/lung_volume)*100
#     print('Vessel to lung area ratio: ', ratio ,"%")
    return vessel_volume, ratio

def split_array_coords(array, indx=0, indy=1):
    x = [array[i][indx] for i in range(len(array))]
    y = [array[i][indy] for i in range(len(array))]
    return x, y


# In[596]:


results.head()


# In[610]:


# K-Means Clustering for Classification of the Lung and Vessel Volumes

from sklearn.cluster import KMeans
xx=[]
yy=[]
zz=[]
for i in range(len(results['Lung Volumes'])):
    x=results['Lung Volumes'][i]
    xx.append(x)
    y=results['Vessel Volumes'][i]
    yy.append(y)

    data2d = np.stack([np.asarray(xx), np.asarray(yy)], axis=1)


kmeans = KMeans(init="random", n_clusters=2)
kmeans.fit(data2d)
print('Centers 2D:', kmeans.cluster_centers_)
print('Slice labels:', kmeans.labels_)

class1 = [data2d[i, :] for i in range(len(kmeans.labels_)) if kmeans.labels_[i] == 0]
class2 = [data2d[i, :] for i in range(len(kmeans.labels_)) if kmeans.labels_[i] == 1]

plt.figure()

x, y = split_array_coords(class1)
plt.scatter(x, y, c='r')
x, y = split_array_coords(class2)
plt.scatter(x, y, c='b')
plt.title('K-means clustered data')
plt.savefig('kmeans-clustered')

plt.figure()
for i in range(len(results['Vessel-Lung Ratio'])):
    z=results['Vessel-Lung Ratio'][i]
    zz.append(z)

plt.scatter(zz, np.arange(len(zz)))
plt.title('Mean Ratios of all Lungs')
plt.savefig('./Ratios')
# plt.close()
plt.show()

# assign categories
categories = np.zeros(len(x), dtype=int)

for c, i in enumerate(x):
    if i > 6:
        categories[c] = int(1)

colormap = np.array(['r', 'b'])

plt.figure()
plt.scatter(np.arange(len(x)) + 1, x, c=colormap[categories])
plt.title('clustered Ratios')
plt.savefig('./clustered_ratios')
plt.show()
plt.close()


# In[ ]:




