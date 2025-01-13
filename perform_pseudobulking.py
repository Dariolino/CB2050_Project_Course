#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 15:55:44 2024

@author: emma (& adjusted by dario)

This is the original script from Emma Gerrits (PhD). This script is used for
pseudobulking the clusters on the Stereo-seq spatial grid for downstream
analysis. The only adjustment made was the changing the input data sets, and
the resolution value. 
"""

#%% import necessary tools
import pandas as pd
import os

#%%
singlenuc_sample='datasetIB5219CB'
spatial_sample = 'Combined_coordinate_file dario'
r=20
os.makedirs('output/03. Pseudobulk/03.1. Count files/' + spatial_sample, exist_ok=True)

#%% load counts under the mask
clusters = [0,5, 8,11, 12]

df = pd.read_csv('output/02. Images/02.2. Cleaned prediction/' + spatial_sample + '/clusters/cluster' + str(clusters[0]) + '_from_' + singlenuc_sample + '_mean_r' + str(r) + '_countsundermask.csv')
df.columns = ['geneID', 'cluster' + str(clusters[0])]

for cluster in clusters[1:]:
    df2 = pd.read_csv('output/02. Images/02.2. Cleaned prediction/' + spatial_sample + '/clusters/cluster' + str(cluster) + '_from_' + singlenuc_sample + '_mean_r' + str(r) + '_countsundermask.csv')
    df2.columns = ['geneID', 'cluster' + str(cluster)]
    df = df.merge(df2, on = 'geneID', how='outer')

#%% load background signal (noise)
noise = pd.read_csv('output/02. Images/02.2. Cleaned prediction/' + spatial_sample + '/clusters/unassigned_from_' + singlenuc_sample + '_mean_r' + str(r) + '_countsundermask.csv')
noise.columns = ['geneID', 'noise']
df = df.merge(noise, on = 'geneID', how = 'outer')
df = df.fillna(0)
df = df.set_index('geneID')

#%% substract noise from the counts
df = df.iloc[:, :(len(clusters)+1)].sub(df['noise'], axis=0)
df[df < 0] = 0
df = df.drop(columns=df.columns[-1])
df = df[(df != 0).any(axis=1)]

#%% rename columns to celltype IDs
df.columns = ['Neurons', 'Purkinje_cells', 'Astrocytes', 'Microglia','Oligodendrocytes']

#%% # normalize data
cpm_df = df.div(df.sum(), axis=1) * 1e6
cpm_df.to_csv('output/03. Pseudobulk/03.1. Count files/' + spatial_sample + '/cpm_file.csv.gz')
