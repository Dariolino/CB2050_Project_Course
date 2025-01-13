#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 10:15:32 2023

@author: emma (& adjusted by dario)

This script is based on the original script from Emma Gerrits (PhD). It serves
to refine the cluster prediction in the Stereo-seq data. The only
adjustment made was changing the input data sets, the clusters to be analyzed, 
the resolution value, and the color scheme file. 
"""

#%%
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
import tifffile as tiff
from tqdm import tqdm
from scipy import ndimage
import matplotlib.pyplot as plt
import os
from PIL import Image

#%%
singlenuc_sample='datasetIB5219CB'
spatial_sample = 'Combined_coordinate_file dario'
clusters = [0,5, 8,11,12]
r=20

#%%
make_images(singlenuc_sample='datasetIB5219CB',
            spatial_sample = 'Combined_coordinate_file dario',
            r=20,
            clusters = [0,5, 8,11, 12],
            colors = pd.read_csv("color_scheme_adjusted.csv", sep = ';'),
            )

#%%
def make_images(singlenuc_sample, spatial_sample, r, clusters, colors):
    
    # make output folders
    os.makedirs('output/05. Combined images/' + spatial_sample, exist_ok=True)

      
    for cluster in clusters:
        print(cluster)
        csv_file = 'output/02. Images/02.2. Cleaned prediction/' + spatial_sample + '/clusters/cluster' + str(cluster) + '_from_' + singlenuc_sample + '_mean_r' + str(r) + '_imagedataframe.csv.gz'
        bf = pd.read_csv(csv_file, compression="gzip")
       
        # create a new image for first cluster
        if cluster==clusters[0]:        
            print("Create new image")
            full_width= int(bf['x'].max()) + 50
            full_height= int(bf['y'].max()) + 50
            new_image = Image.new("RGBA", (full_width+1, full_height+1), (255, 255, 255, 0))

        # extract relevant columns from the DataFrame
        normalized_values = bf['pixelintensity'].values
        x_values = bf['x'].astype(int).values
        y_values = bf['y'].astype(int).values

        # positive and negative values separation
        positive_mask = normalized_values >= 0
        positive_values = normalized_values[positive_mask]
        positive_values_scaled = positive_values / positive_values.max()
        transparency_values = [max(100, int(255 * intensity)) for intensity in positive_values_scaled]

        if cluster==clusters[0]: 
            pixels = new_image.load()        

        rgb_color = colors.iloc[cluster,:]
        rgb_color = tuple(rgb_color.loc[['R', 'G', 'B']])
        rgb_color = tuple(int(x) for x in rgb_color)

        for x, y, transparency in zip(x_values[positive_mask], y_values[positive_mask], transparency_values):
            # set RGB values with constant color and varying transparency
            pixels[x, y] = rgb_color + (transparency,)

        bf2 = bf.loc[:,["x", "y", "bin1_ID", "pixelintensity"]]
    
        
    new_image.save('output/05. Combined images/' + spatial_sample + '/Overlay_from_' + singlenuc_sample + '_mean_r' + str(r) + '.png')