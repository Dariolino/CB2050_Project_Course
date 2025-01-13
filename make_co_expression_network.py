#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 15:41:40 2023

@author: emma (& adjusted by Dario)

This script is based on the original script from Emma Gerrits (PhD). It is used
to preprocess the snRNA-seq data and derive cell type-specific coexpression 
sets. The only adjustment made was changing the input data sets, adding known 
cerebellar marker genes, and optimizing the parameters for DBSCAN clustering.
"""

import numpy as np
import pandas as pd
import scanpy as sc
import umap.umap_ as umap
import matplotlib.pyplot as plt
from scipy.stats import zscore
from tqdm import tqdm
from numpy import where
from sklearn.cluster import DBSCAN
from matplotlib import pyplot
import os


#%% read in single-nuc data
sample = "datasetIB5219CB"
sub_path = "Task_2/Cell_segmentation_scripts/"
adata = sc.read_h5ad('input/' + sample + ".h5ad")

# extract raw countmatrix
df = pd.DataFrame(adata.raw.X.toarray(), index=adata.obs.index, columns=adata.var.index)

os.makedirs('output/01. Co-expression network/' + sample, exist_ok=True)

#%% gene filtering
col_sums = (df != 0).sum(axis=0)
#col_sums = (df >= 2).sum(axis=0)

# export list of the genes with total counts
col_sums.to_csv('output/01. Co-expression network/' + sample + '/genes.csv')

# filter for genes expressed in at least 50 of the cells
#sc.pp.filter_genes(adata, min_cells=50)
df = df.loc[:, col_sums >= 50]


#%% generating count table
# only include cells with at least 2 counts of a gene
greater_than_1 = df >= 2
row_indices = [np.where(greater_than_1[col])[0] for col in df.columns]

# subset table for geneX-positive cells and sum the counts 
coexpression_matrix = np.zeros((len(row_indices), df.shape[1]), dtype=float)
with tqdm(total=len(row_indices), leave=True) as pbar:
    for i, indices in enumerate(row_indices):
        if len(indices) > 0:
            sub = df.values[indices, :]
            coexpression_matrix[i,:] = (sub.sum(axis=0) / len(indices))
        pbar.update(1)  
pbar.close()        
        
coexpression_matrix = pd.DataFrame(coexpression_matrix) # rows are samples, columns are genes
coexpression_matrix.columns = df.columns
coexpression_matrix.index = df.columns

#coexpression_matrix.to_csv('./Results/01. Co-expression matrix/' + sample + '/coexpression_matrix_mean.csv', compression='gzip')
#%%
# zscore by column
coexpression_matrix_scaled = zscore(coexpression_matrix, axis=0, nan_policy='omit') # by column
coexpression_matrix_scaled = coexpression_matrix_scaled.dropna(how='all', axis = 0)

coexpression_matrix_scaled.to_csv('output/01. Co-expression network/' + sample + '/coexpression_matrix_scaled_notT.csv.gz', compression='gzip')

# columns are cells, rows are genes
coexpression_matrix_scaled = coexpression_matrix_scaled.T 

# export z-score table
genes = pd.DataFrame(coexpression_matrix_scaled.index)
genes.to_csv('output/01. Co-expression network/' + sample + '/coexpression_matrix_scaled_index.csv')

coexpression_matrix_scaled.to_csv('output/01. Co-expression network/' + sample + '/coexpression_matrix_scaled.csv.gz', compression='gzip')

#%% Run UMAP
# create a UMAP reduction
# uses the rows as samples!!!
reducer = umap.UMAP(
        n_neighbors=15,#15
        min_dist=0.3, # 0.3
        n_components=2,
        metric='euclidean',
        n_epochs=100, # 100
        spread = 3.0, #3
        random_state=42
    )

# fit and transform your data using UMAP
umap_result = reducer.fit_transform(coexpression_matrix_scaled)

#%%
# visualize the UMAP result
plt.scatter(umap_result[:, 0], umap_result[:, 1], s=0.8, c="black", alpha=0.1, linewidths=0)
plt.title('UMAP Visualization') 
plt.show()


#%% another plot
coords_df = pd.DataFrame(umap_result, columns=['x', 'y'])
coords_df.index = coexpression_matrix_scaled.index

# subset the data for labels
label_subset = [
    "GFAP", "AQP4", "ALDH1L1",   "PLP1",  "MBP", "OPALIN", "WDR49", "CFAP47", "CP",
    "P2RY12", "TMEM119", "CSF1R", "CKB",
     "GAD2", "SLC17A7", "MAP2", "SNAP25", "CARTPT", "NPY", "SST", "GAD1", "SLC17A6", "RORB", "CUX2",
     "GAPDH", "HMGB1", "HPRT1", "MALAT1", "NEAT1", "CLDN5", "FLT1",
     "PPP1R17","PVALB", "HOMER3", "CA8", "NPTX1", "PCP4", "CALB1", "ITPR1", "SLC1A6", "PDGFRB", "ACTA2", "MYH11", "DCN", "COL1A1", "IL7R"
]



label_subset = []
purkinje_genes = ["PPP1R17","PVALB", "HOMER3", "CA8", "PCP4", "CALB1", "ITPR1", "SLC1A6"]
astrocyte_genes = ["S100B", "ALDH1L1", "AQP4", "SLC1A2", "SOX9"] # astrocyte markers #Bergmann glia has SLC1A3 and SOX2?# very similar to astrocytes since they are a specialized form of them 
microglia_genes = ["P2RY12", "CX3CR1", "ITGAM", "PTPRC"] # cluster 11
oligodendrocyte_genes = ["MBP", "MOG", "PLP1", "OLIG2", "SOX10", "CNP", "MAG"] # all in cluster 12          # "NKX2-2" is in cluster 8! # "OLIG1" not in cluster



# filtered dataframe for labels
label_df = coords_df[coords_df.index.isin(astrocyte_genes)]

# scatter plot
plt.scatter(coords_df['x'], coords_df['y'], s=0.5, color='grey', alpha=0.1, linewidths=0)

# scatter plot with labels
plt.scatter(label_df['x'], label_df['y'], s=0.5, color='red', label="Labels")


#
label_df_purkinje = coords_df[coords_df.index.isin(purkinje_genes)]
label_df_astro = coords_df[coords_df.index.isin(astrocyte_genes)]
label_df_microglia = coords_df[coords_df.index.isin(microglia_genes)]
label_df_oligodendrocyte = coords_df[coords_df.index.isin(oligodendrocyte_genes)]

# add labels with text adjustment
#texts = [plt.text(x, y, label) for label, x, y in zip(label_df.index, label_df['x'], label_df['y'])]

# adjust the positions of labels to avoid overlaps
#adjust_text(texts, arrowprops=dict(arrowstyle='->', color='red'), force_text=(0.1, 0.1))

#plt.savefig("./output/01. Co-expression network/" + sample + "/UMAP_geneannotation" + ".pdf", format = "pdf", transparent = True)

plt.show()


#%% dbscan clustering

import matplotlib.lines as mlines

# define dataset and model
X = umap_result
model = DBSCAN(eps=0.72, min_samples=75) # 0.52, 25
# fit model and predict clusters
yhat = model.fit_predict(X)
clusters = np.unique(yhat)
# define colormap
colors = plt.cm.rainbow(np.linspace(0, 1, len(clusters)))

# make scatter plot for samples from each cluster
for cluster, color in zip(clusters, colors):
 row_ix = where(yhat == cluster) # get row indexes for samples with this cluster
 plt.scatter(X[row_ix, 0], X[row_ix, 1], s=0.2, linewidth=0, label=f'Cluster {cluster}', c=[color])
 # calculate the cluster center
 cluster_center = np.mean(X[row_ix], axis=0)
 # label the center with the cluster number
 plt.text(cluster_center[0], cluster_center[1], str(cluster), fontsize=8, color='black', ha='center', va='center')
 plt.xlabel('UMAP Dimension 1') 
 plt.ylabel('UMAP Dimension 2') 
 plt.scatter(label_df['x'], label_df['y'], s=1, color='red', label="Labels") # for red dots

 plt.scatter(label_df_purkinje['x'], label_df_purkinje['y'], s=1, color='darkblue', label="Purkinje cell markers") # for purkinje darkblue dots
 plt.scatter(label_df_astro['x'], label_df_astro['y'], s=1, color='darkgreen', label="Astrocyte markers") # for astrocyte drark green dots
 plt.scatter(label_df_microglia['x'], label_df_microglia['y'], s=1, color='brown', label="Microglia markers") # for microglia brown dots  
 plt.scatter(label_df_oligodendrocyte['x'], label_df_oligodendrocyte['y'], s=1, color='red', label="Oligodendrocyte markers") # for oligodendrocyte red dots

# add the legend
dark_blue_dot = mlines.Line2D([], [], color='darkblue', marker='o', linestyle='None', markersize=5, label='Purkinje cell markers')
dark_green_dot = mlines.Line2D([], [], color='darkgreen', marker='o', linestyle='None', markersize=5, label='Astrocyte markers')
brown_dot = mlines.Line2D([], [], color='brown', marker='o', linestyle='None', markersize=5, label='Microglia markers')
red_dot = mlines.Line2D([], [], color='red', marker='o', linestyle='None', markersize=5, label='Oligodendrocyte markers')
plt.legend(handles=[dark_blue_dot, dark_green_dot, brown_dot, red_dot], loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')

# adjust the layout
plt.tight_layout(rect=[0, 0, 1, 0.95])  
plt.savefig("./output/01. Co-expression network/" + sample + "/UMAP_geneclustering" + ".pdf", format = "pdf", transparent = True)
pyplot.show()


#%% extract zscore lists per cluster
coords_df['cluster'] = yhat
coords_df.to_csv('./output/01. Co-expression network/' + sample + '/Cluster_annotation.csv')

coexpression_matrix_scaled.columns = df.columns
coexpression_matrix_scaled.index = df.columns

for cluster in clusters:
    row_ix = where(yhat == cluster)
    coexpression_matrix_scaled_sub = coexpression_matrix_scaled.iloc[row_ix] 
    result_mean = coexpression_matrix_scaled_sub.mean(axis = 0)
    result_sum = coexpression_matrix_scaled_sub.sum(axis = 0)
    
    result_df = pd.DataFrame({'mean': result_mean, 'sum': result_sum})
    result_df.to_csv('output/01. Co-expression network/' + sample + '/Cluster' + str(cluster) + '.csv')
