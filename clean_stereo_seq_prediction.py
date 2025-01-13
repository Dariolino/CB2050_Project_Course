#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 09:44:45 2024

@author: emma (& adjusted by dario)

This script is based on the the original script from Emma Gerrits (PhD). This
script refines the prediction of snRNA-seq derived co-expression clusters onto
the Stereo-seq spatial grid. The only adjustment made was changing the input
data sets, and the resolution value. 
"""

#%% import necessary tools
import pandas as pd
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
from PIL import Image
import os


#%%
singlenuc_sample='datasetIB5219CB'
spatial_sample = 'Combined_coordinate_file dario'
r=20
method='stereoseq'

#%%
os.makedirs('output/02. Images/02.2. Cleaned prediction/' + spatial_sample + '/clusters/', exist_ok=True)

#%% load gem file
if method=='stereoseq':
    csv_file = 'input/' + spatial_sample + '.csv' 
    df = pd.read_csv(csv_file)
    df = df.rename(columns={'bin_ID': 'bin1_ID'})
    df = df.rename(columns={'MIDCount': 'UMICount'})
    print(df)
if method=='merfish':
    df = pd.read_csv('input/' + spatial_sample + '.barcodes.csv')
    df['bin1_ID'] = df.index
    df = df.drop(columns=df.columns[[0,1, 5, 6]])
    df.columns = ['geneID', 'x', 'y', 'bin1_ID']
    df['MIDCount'] = 1
    df['x'] = df['x'] - df['x'].min()
    df['y'] = df['y'] - df['y'].min()
    df['z'] = 0
    df = df.rename(columns={'MIDCount': 'UMICount'})
if method=='xenium':
    csv_file = 'input/' + spatial_sample + '_transcripts.csv'
    df = pd.read_csv(csv_file)
    df = df.drop(columns=df.columns[[1, 2, 7, 8,9]])
    df.columns = ['bin1_ID','geneID', 'x', 'y', 'z']
    df['MIDCount'] = 1  
    df = df.rename(columns={'MIDCount': 'UMICount'})


full_x=int(round(df['x'].max(), 0))
full_y=int(round(df['y'].max(), 0))

print("full x is: ", full_x)

print("full y is: ", full_y)
#full_x = 21010-1
#full_y = 20458-1
gemfile = df
print("Gemfile is: ", gemfile)
#%%
clusters = [0,4,5,11,13]
clusters = [9,8,21,4,2]

r = 20
clusters = [0,5,8,11,12]
for cluster in clusters:
    print(cluster)
    csv_file = 'output/02. Images/02.1. Raw prediction/' + spatial_sample + '/clusters/cluster' + str(cluster) + '_from_' + singlenuc_sample + '_mean_r' + str(r) + '_imagedataframe.csv.gz'
    df = pd.read_csv(csv_file, compression='gzip')
    df = df.loc[:, ['bin1_ID', 'block_sum']]
    df = df.rename(columns={'block_sum': 'cluster' + str(cluster)})
    df = df.drop_duplicates()
    print(df)
    gemfile = gemfile.merge(df, how='left', on='bin1_ID')
    
#%% this step takes a while
gemfile_sub = gemfile.iloc[:, 8:len(gemfile.columns)]
print(gemfile_sub)

#%% 
#gemfile['annotation'] = gemfile_sub.apply(
    #lambda row: np.nan if row.isna().all() or row.eq(0).all() else row.idxmax(), 
    #axis=1)


# faster approach
# create a mask for rows where all values are NaN or all values are 0
all_na_mask = gemfile_sub.isna().all(axis=1)
all_zero_mask = gemfile_sub.eq(0).all(axis=1)

# use idxmax to get the index of the maximum value (ignoring NaN by default)
idxmax_values = gemfile_sub.idxmax(axis=1)

# combine the masks to assign NaN where conditions are met, otherwise use idxmax
gemfile['annotation'] = np.where(all_na_mask | all_zero_mask, np.nan, idxmax_values)


gemfile.to_csv('output/02. Images/02.2. Cleaned prediction/' + spatial_sample + '/clusters/annotationfile_from_' + singlenuc_sample + '_mean_r' + str(r) + '_gemfile_with_annotations.csv.gz', compression='gzip')


# alternative script
# Dynamically detect cluster columns (columns that start with 'cluster')
# cluster_columns = [col for col in gemfile.columns if col.startswith('cluster')]

# Check if there's only one cluster column or multiple
# if len(cluster_columns) == 1:
#     # If only one cluster column, annotate directly
#     single_cluster = cluster_columns[0]
#     print(f"Only one cluster detected: {single_cluster}")
#     gemfile['annotation'] = np.where(
#         gemfile[single_cluster].isna() | (gemfile[single_cluster] == 0),
#         np.nan,  # Assign NaN if the cluster value is NaN or 0
#         single_cluster  # Otherwise, assign the annotation as the cluster name
#     )
# else:
#     # If multiple cluster columns, calculate annotation based on the highest score
#     print(f"Multiple clusters detected: {cluster_columns}")
#     gemfile_sub = gemfile[cluster_columns]
    
#     gemfile['annotation'] = gemfile_sub.apply(
#         lambda row: np.nan if row.eq(0).all() else row.idxmax(), axis=1
#     )

# # Save the updated gemfile with annotations
# gemfile.to_csv(
#     f'output/02. Images/02.2. Cleaned prediction/{spatial_sample}/clusters/annotationfile_from_{singlenuc_sample}_mean_r{r}_gemfile_with_annotations.csv.gz',
#     compression='gzip')

#%% identify pixels that are not under any of the masks
unassigned_mask = gemfile[gemfile_sub.isna().all(axis=1)]
unassigned_mask_counts = pd.DataFrame(unassigned_mask.groupby(by=["geneID"])["UMICount"].sum() / len(unassigned_mask) * 1e6) 
unassigned_mask_counts.to_csv('output/02. Images/02.2. Cleaned prediction/' + spatial_sample + '/clusters/unassigned_from_' + singlenuc_sample + '_mean_r' + str(r) + '_countsundermask.csv')

#%% make image of the unassigned mask (just to see what it looks like)
new_image = np.zeros((full_x+1, full_y+1, 3), dtype=np.uint8)
x_values = unassigned_mask['x'].astype(int).values - 1
y_values = unassigned_mask['y'].astype(int).values - 1
new_image[x_values, y_values,  1] = 255

# save the image
tiff.imsave('output/02. Images/02.2. Cleaned prediction/' + spatial_sample + '/clusters/unassigned_from_' + singlenuc_sample + '_mean_r' + str(r) + '.tif', new_image)

#%% make new images and counts for each cluster
outputimageformat='png' # png or tiff. png has white background, tiff has black background

cap = 1000
for cluster in clusters:
    print(cluster)
    result_df = gemfile[gemfile['annotation'] == 'cluster' + str(cluster)]
    result_df['cluster' + str(cluster)][result_df['cluster' + str(cluster)] > cap] = cap
    result_df['x'] = result_df['x'] - 1
    result_df['y'] = result_df['y'] - 1
    
    # make image
    new_image = np.zeros((full_x+1, full_y+1, 3), dtype=np.uint8)
    max_sum = result_df['cluster' + str(cluster)].max()
    min_sum = result_df['cluster' + str(cluster)].min()*-1
    normalized_values = result_df['cluster' + str(cluster)].values
    x_values = result_df['x'].astype(int).values
    y_values = result_df['y'].astype(int).values
    
    if outputimageformat=='tiff':
        print("Create tiff image")
        positive_mask = normalized_values >= 0
        negative_mask = ~positive_mask
        positive_values = normalized_values[positive_mask]
        negative_values = -normalized_values[negative_mask]
        new_image[x_values[positive_mask], y_values[positive_mask],  1] = (positive_values * (255 / max_sum)).astype(int)
        new_image[x_values[negative_mask], y_values[negative_mask], 2] = (negative_values * (255 / min_sum)).astype(int)
        tiff.imsave('output/02. Images/02.2. Cleaned prediction/' + spatial_sample + '/clusters/cluster' + str(cluster) + '_from_' + singlenuc_sample + '_mean_r' + str(r) + '_cap' + str(cap) + '.tif', new_image)
    
    # make png image with white background
    if outputimageformat=='png':
        print("Create png image")
        new_image = Image.new("RGBA", (full_x+1, full_y+1), (255, 255, 255, 0))
        positive_mask = normalized_values >= 0
        positive_values = normalized_values[positive_mask]
        positive_values_scaled = positive_values / positive_values.max()
        transparency_values = [max(100, int(255 * intensity)) for intensity in positive_values_scaled]
        pixels = new_image.load() 
        rgb_color = (255, 0, 0) # red 
        for x, y, transparency in zip(x_values[positive_mask], y_values[positive_mask], transparency_values):
            pixels[x, y] = rgb_color + (transparency,)
        new_image.save('output/02. Images/02.2. Cleaned prediction/' + spatial_sample + '/clusters/cluster' + str(cluster) + '_from_' + singlenuc_sample + '_mean_r' + str(r) + '_cap' + str(cap) + '.png')

    # save tables
    mask_counts = pd.DataFrame(result_df.groupby(by=["geneID"])["UMICount"].sum() / len(result_df) * 1e6) 
    mask_counts.to_csv('output/02. Images/02.2. Cleaned prediction/' + spatial_sample + '/clusters/cluster' + str(cluster) + '_from_' + singlenuc_sample + '_mean_r' + str(r) + '_countsundermask.csv')

    plt.figure()
    fix, axs = plt.subplots(1, 1, figsize =(10,7), tight_layout = True)
    axs.hist(result_df['cluster' + str(cluster)], bins = 100)
    plt.axvline(x = 0, color = 'r')
    plt.savefig('output/02. Images/02.2. Cleaned prediction/' + spatial_sample + '/clusters/cluster' + str(cluster) + '_from_' + singlenuc_sample + '_mean_r' + str(r) + '_histogram.pdf', format = 'pdf')

    # this is the dataframe that makes up the final image
    result_df = result_df[['x', 'y', 'bin1_ID', 'UMICount', 'geneID', 'cluster' + str(cluster)]]
    result_df.columns = ['x', 'y', 'bin1_ID', 'UMICount', 'geneID', 'pixelintensity']
    result_df.to_csv('output/02. Images/02.2. Cleaned prediction/' + spatial_sample + '/clusters/cluster' + str(cluster) + '_from_' + singlenuc_sample + '_mean_r' + str(r) + '_imagedataframe.csv.gz', compression='gzip')
