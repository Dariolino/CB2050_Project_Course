#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 18:20:23 2024

@author: emma & dario

This is based on the original script from Emma Gerrits (PhD). 
The script was adjusted to be able to plot multiple genes in expression bar 
plots at the same time. Furthermore, this script entails a function for 
filtering the Stereo-seq data to find genes enriched in Purkinje cells, and
plotting respective gene expression data as bar plots. 

In the second part of this script, a list of genes specific to the cerebellum
(derived from the Human Protein Atlas) is compared to the list of Purkinje cell
enriched genes.
"""

#%% import necessary tools
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
import numpy as np
import json 


#%%
singlenuc_sample="datasetIB5219CB"
spatial_sample = "Combined_coordinate_file dario"
r=20
os.makedirs("output/03. Pseudobulk/03.2. Barplots/" + spatial_sample,
             exist_ok=True)

cpm_df = pd.read_csv("output/03. Pseudobulk/03.1. Count files/" +
                      spatial_sample + "/cpm_file.csv", index_col=0)

#%% load color scale
color_scale_df = pd.read_csv("color_scheme_adjusted.csv", index_col=0, sep=";")
color_scale_df = color_scale_df.set_index("Cell_type")

#%% make PCA plot (not necessary to include, was just curious)
logCPM_df = cpm_df + 1
logCPM_df = logCPM_df.apply(np.log)
pca = PCA(n_components=2)
pca.fit(logCPM_df.T)
principal_components = pca.transform(logCPM_df.T)
plt.figure(figsize=(8, 6))
plt.scatter(principal_components[:, 0], principal_components[:, 1])
for i, name in enumerate(logCPM_df.columns):
    plt.annotate(name, (principal_components[i, 0], principal_components[i, 1]))
plt.title("PCA of Gene Expression Data")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True)
plt.show()

#%% make gene expression plots
def plot_gene_expression(ax, df, gene, color_scale_df = None,
                          include_legend = True):
    """Plot gene expression data across celltypes as a bar plot"""

    # check if the gene is present in the dataframe
    if gene not in df.index.values:
        print(f"The gene '{gene}' was not detected in the spatial data.")
        return None
    
    gene_expression = df.loc[gene]

    if color_scale_df is not None:
        for i, (sample, expr_value) in enumerate(gene_expression.items()):
            if sample in color_scale_df.index:
                color = color_scale_df.loc[sample]["HEX"]
                ax.bar(i, expr_value, color=color, label=sample)
            else:
                print(f"The sample '{sample}' was not found in color_scale_df.")
    else:
        ax.bar(range(len(gene_expression)), gene_expression, color="skyblue")

    ax.set_title(f"Gene Expression of {gene}", fontsize=16, weight = "bold") 
    ax.set_xlabel("Cell type masks", fontsize=16)
    ax.set_ylabel("Noise-corrected CPM", fontsize=16)
    ax.set_xticks(range(len(gene_expression)))
    ax.set_xticklabels(gene_expression.index, rotation=45, fontsize = 14)
    if include_legend:
        ax.legend()


def plot_multiple_genes(genes, df, color_scale_df, n_rows=3, n_cols=4,
                         figsize=(20, 10), include_legend=True):
    """Plot bar plots for multiple genes in a single figure"""

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    for ax, gene in zip(axes, genes):
        plot_gene_expression(ax, df, gene, color_scale_df, include_legend)

    plt.tight_layout()
    plt.show()


def get_maximally_expressed_genes_with_plots( # we could modularize this function by making separate functions for the plots
    df, celltype, min_fold_change_to_avg_of_other_celltypes=2, 
    min_fold_change_to_2nd_highest_celltype=1.65, min_cpm_expr=10):
    """Filter genes based on expression in a specific cell type and plot the
       resulting gene expression data as bar plots

       Filter 1: Genes in which Purkinje cells have the highest expression
       Filter 2: Genes with more than the minimum CPM expression (default is 10)
       Filter 3: Genes for which the Purkinje cell expression is higher than 
                 for the mean expression across the other celltypes times a factor
       Filter 4: Genes for which the Purkinje cell expression is higher than
                 for the 2nd highest expressingcell type times a factor
    
       Outputted plots:
       plot 1: CPM vs. fold change to mean expression across cerebellar cell types
       plot 2: CPM vs. fold change to 2nd highest expression across cell types
       plot 3: fold change to mean vs. fold change to 2nd highest expression
    """
    
    print("Number of genes before filtering: ", df.shape[0])

    # filter genes where Purkinje cells have the highest CPM values
    highest_expression_genes = df.index[df[celltype] == df.max(axis=1)].tolist()
    print("Filter 1: Number of genes where the specified cell type has the" +
          " highest expression: ", len(highest_expression_genes) - 1)
    
    # store genes which pass additional filters in a list
    super_highest_expressed_genes = []
    filter_min_cpm_count, filter_avg_count, filter_2nd_highest_count = 0, 0, 0
    plot_data = {"Gene_Name": [], "Purkinje_CPM": [], "Fold_Change_Mean": [], 
                 "Fold_Change_2nd_Highest": [], "Filtered": []}

    for gene in highest_expression_genes:
        celltype_value = df.loc[gene, celltype]
        other_values = df.loc[gene].drop(celltype)
        mean_other_values = other_values.mean()
        second_highest = other_values.max()
        
        # the small decimal is used to avoid taking the log of 0
        fold_change_mean = np.log2(celltype_value / (mean_other_values + 0.0000000000001))
        fold_change_2nd_highest = np.log2(celltype_value / (second_highest + 0.0000000000001))
        
        # collect the plot data
        plot_data["Gene_Name"].append(gene)
        plot_data["Purkinje_CPM"].append(celltype_value)
        plot_data["Fold_Change_Mean"].append(fold_change_mean)
        plot_data["Fold_Change_2nd_Highest"].append(fold_change_2nd_highest)
        plot_data["Filtered"].append(False)  
        
        # this changes the filtering status based on the filtering criteria
        if celltype_value >= min_cpm_expr:
            filter_min_cpm_count += 1
            if celltype_value >= min_fold_change_to_avg_of_other_celltypes * mean_other_values:
                filter_avg_count += 1
                if celltype_value >= min_fold_change_to_2nd_highest_celltype * second_highest:
                    filter_2nd_highest_count += 1
                    super_highest_expressed_genes.append(gene)
                    plot_data["Filtered"][-1] = True

    print(f"Filter 2: Remaining number of genes with CPM >= {min_cpm_expr}: ",
           filter_min_cpm_count)
    print("Filter 3: Remaining number of genes with fold change >=" +
          f" {min_fold_change_to_avg_of_other_celltypes} to the average: ",
            filter_avg_count)
    print("Filter 4: Remaining number of genes with fold change >= " +
          f"{min_fold_change_to_2nd_highest_celltype} to the 2nd highest: ",
          filter_2nd_highest_count)

    # plot 1: CPM vs. fold change to mean expression across cerebellar cell types
    plt.figure(figsize=(10, 6))
    for i in range(len(plot_data["Purkinje_CPM"])):
        color = "red" if plot_data["Filtered"][i] else "gray"
        plt.scatter(plot_data["Fold_Change_Mean"][i],
                    plot_data["Purkinje_CPM"][i], color=color, alpha=0.7)

    plt.axvline(np.log2(min_fold_change_to_avg_of_other_celltypes),
                color="blue", linestyle="--",
                label=f"Mean Fold Change Threshold (log2({min_fold_change_to_avg_of_other_celltypes}))")
    plt.axhline(min_cpm_expr, color="green", linestyle="--",
                label=f"CPM Threshold ({min_cpm_expr})")
    plt.title("Purkinje Cell Expression vs. Fold Change to Mean")
    plt.xlabel("Log2 Fold Change (Purkinje vs. Mean Other Cells)")
    plt.ylabel(f"{celltype} CPM")
    plt.legend()
    plt.show()

    # plot 2: CPM vs. fold change to 2nd highest expression across cell types
    plt.figure(figsize=(10, 6))
    for i in range(len(plot_data["Purkinje_CPM"])):
        color = "red" if plot_data["Filtered"][i] else "gray"
        plt.scatter(plot_data["Fold_Change_2nd_Highest"][i],
                    plot_data["Purkinje_CPM"][i], color=color, alpha=0.7)

    plt.axvline(np.log2(min_fold_change_to_2nd_highest_celltype),
                color="purple", linestyle="--",
                label=f"2nd Highest Fold Change Threshold (log2({min_fold_change_to_2nd_highest_celltype}))")
    plt.axhline(min_cpm_expr, color="green", linestyle="--",
                label=f"CPM Threshold ({min_cpm_expr})")
    plt.title("Purkinje Cell Expression vs. Fold Change to 2nd Highest")
    plt.xlabel("Log2 Fold Change (Purkinje vs. 2nd Highest Expression)")
    plt.ylabel(f"{celltype} CPM")
    plt.legend()
    plt.show()

    # plot 3: fold change to mean vs. fold change to 2nd highest expression
    plt.figure(figsize=(10, 6))
    for i in range(len(plot_data["Purkinje_CPM"])):
        if plot_data["Purkinje_CPM"][i] >= min_cpm_expr:
            color = "red" if plot_data["Filtered"][i] else "gray"
            plt.scatter(plot_data["Fold_Change_Mean"][i],
                        plot_data["Fold_Change_2nd_Highest"][i], color=color,
                        alpha=0.7)

    plt.axvline(np.log2(min_fold_change_to_avg_of_other_celltypes),
                color="blue", linestyle="--",
                label=f"Mean Fold Change Threshold (log2({min_fold_change_to_avg_of_other_celltypes}))")
    plt.axhline(np.log2(min_fold_change_to_2nd_highest_celltype),
                color="purple", linestyle="--",
                label=f"2nd Highest Fold Change Threshold (log2({min_fold_change_to_2nd_highest_celltype}))")
    plt.title("Fold Change to Mean vs. Fold Change to 2nd Highest")
    plt.xlabel("Log2 Fold Change (Purkinje vs. Mean Other Cells)")
    plt.ylabel("Log2 Fold Change (Purkinje vs. 2nd Highest)")
    plt.legend()
    plt.show()

    # the plot data is returned to allow for detailed downstream analysis
    return super_highest_expressed_genes, plot_data

def create_subset_gene_lists(genes, subset_size=12, subset_corrector=3):
    """Create subsets of genes for plotting multiple genes in a single figure"""
    subsets = []
    for i in range(0, len(genes), subset_size):
        subset = genes[i:i + subset_size]
        if len(subset) < subset_size:
            subset += genes[-subset_corrector:]
        subsets.append(subset)
    return subsets

#%% 

# exemplary gene lists
genes = ["MAP1B", "AQP4", "CA8","ITGAM", "FLT1", "GAPDH"]

genes_2 = ["AAGAB", "AATF", "ABAT", "ABCA12", "ABCA13", "ABCA3", "ABCA7",
           "ABCG8", "ABHD3", "ABHD6", "ABLIM2", "ACAD8", "ACAD9", "ACAT1"]

genes_3 = ["ACTBP15", "ACTG1P23", "ACTG1P9", "ADAM3B", "ADORA2BP1", "AGGF1P3",
           "ANKRD18EP", "ANKRD20A19P", "ANKRD33BP8", "AP2B1P1", "ARHGAP26-IT1",
           "ARIH2P1", "ARL5AP5", "ARPC3P1", "ARPC3P2", "ARPC3P4", "ASS1P1",
           "ASS1P12", "ASS1P13", "ASS1P9"]

genes_4 = ["CALCA", "DMKN", "DSTNP3", "MC4R", "RSU1P3", "SHLD2P3"]

genes_5 = ["PCP4", "PVALB", "CALB1"]

genes_6 = ["PCP4", "PVALB", "CALB1", "HOMER3", "ITPR1", "ITM2B"]

genes_7 = ["PCP4", "PVALB", "CALB1", "HOMER3", "ITPR1", "ITM2B", "GRID2",
           "PCP2", "YWHAH"] # Purkinje cell-enriched genes with the higgest CPM values
 

# exemplary plot for 9 Purkinje cell-enriched genes 
plot_multiple_genes(genes_7, cpm_df, color_scale_df, n_rows = 3, n_cols = 3,
                    figsize = (20,15), include_legend=False)


# %% filter Purkinje cell-enriched genes and plot gene expression data

# this is the main command for deriving Purkinje cell-enriched genes
max_genes, more_data = get_maximally_expressed_genes_with_plots(df = cpm_df,
                              celltype = "Purkinje_cells",
                              min_fold_change_to_avg_of_other_celltypes = 2, 
                              min_fold_change_to_2nd_highest_celltype = 1.65, 
                              min_cpm_expr = 10) 


subsets = create_subset_gene_lists(max_genes, subset_size=12,
                                   subset_corrector = 3) 
print(subsets)

# make bar plots (in batches) for all Purkinje cell-enriched genes
for i, subset in enumerate(subsets):
    plot_multiple_genes(subset, cpm_df, color_scale_df, n_rows=3, n_cols=4,
                         figsize=(20, 12), include_legend= False) 
    #plt.savefig("./output/03. Pseudobulk/03.2. Barplots/" + spatial_sample + 
    # f"/combined_gene_expression_{i}.pdf", format="pdf")
    plt.show()

#subsets_in_three = create_subset_gene_lists(max_genes, subset_size=3)

# save to a text file if needed for further analysis
# with open("Purkinje_gene_list_cpm_10.txt", "w") as file:
#     for gene in max_genes:
#         file.write(gene + "\n")

# %%
# check if Purkinje cell-enriched genes overlap with lists annotated as
# region-enriched, group-enriched or region-enhanced from Human Protein Atlas
# this essentialy check how much genes charactersitic of tthe cerebellum could be attributed to Purkinje cells
# (https://www.proteinatlas.org/search/brain_category_rna%3Acerebellum%3BRegion+enriched+AND+sort_by%3Atissue+specific+score)

# import region enriched, group enriched and region enhanced genes 
hpa_reg_enriched_tb = pd.read_csv("brain_category_rna_cerebellum_region_enriched_64.tsv" , sep="\t") 
hpa_group_enriched_tb = pd.read_csv("brain_category_rna_cerebellum_group_enriched_33.tsv" , sep="\t")
hpa_reg_enhanced_tb = pd.read_csv("brain_category_rna_cerebellum_region_enhanced_65.tsv" , sep="\t")

# extract the gene names from the imported tables from the HPA
hpa_reg_enr_genes = hpa_reg_enriched_tb["Gene"].tolist()
hpa_group_enr_genes = hpa_group_enriched_tb["Gene"].tolist()
hpa_reg_enh_genes = hpa_reg_enhanced_tb["Gene"].tolist()

# match the gene with the genes available in stereo seq data to get the overlap 
hpa_reg_enr_genes_in_stereo, hpa_group_enr_genes_in_stereo = [],[]
hpa_reg_enh_genes_in_stereo = []

for gene in cpm_df.index:
    if gene in hpa_reg_enr_genes:
        hpa_reg_enr_genes_in_stereo.append(gene)
    if gene in hpa_group_enr_genes:
        hpa_group_enr_genes_in_stereo.append(gene)
    if gene in hpa_reg_enh_genes:
        hpa_reg_enh_genes_in_stereo.append(gene)

print("Number of region-enriched genes found in Stereo-seq data: ", 
      len(hpa_reg_enr_genes_in_stereo), "out of", len(hpa_reg_enr_genes))
print("Number of group-enriched genes found in Stereo-seq data: ", 
      len(hpa_group_enr_genes_in_stereo), "out of", len(hpa_group_enr_genes))
print("Number of region-enhanced genes found in Stereo-seq data: ", 
      len(hpa_reg_enh_genes_in_stereo), "out of", len(hpa_reg_enh_genes))

# check if the genes are among characteristic Purkinje cell genes (in max_genes)
reg_enr_purkinje, group_enr_purkinje, reg_enh_purkinje = [], [], []
for gene in max_genes:
    if gene in hpa_reg_enr_genes_in_stereo:
        reg_enr_purkinje.append(gene)
    if gene in hpa_group_enr_genes_in_stereo:
        group_enr_purkinje.append(gene)
    if gene in hpa_reg_enh_genes_in_stereo:
        reg_enh_purkinje.append(gene)

print("The following region-enriched genes are characteristic for Purkinje cells:",
       reg_enr_purkinje)
print("The following group-enriched genes are characteristic for Purkinje cells:",
       group_enr_purkinje)
print("The following region-enhanced genes are characteristic for Purkinje cells:",
       reg_enh_purkinje)

# --> 3 region enriched genes are enriched in Purkinje cells: NRK, SLC1A6, ZNF385C
# --> 3 group enriched genes are enriched in Purkinje cells: GNG13, PCP2. PPP1R17
# --> 3 region enhanced genes are enriched in Purkinje cells: CA8, GRID2, IRF6


# %%
# export Purkinje cell-enriched genes (including their expression data) 
# to R for GO term and KEGG enrichment analysis

with open("Purkinje_genes_detailed.json", "w") as outfile:
    json.dump(more_data, outfile)

with open("Purkinje_genes.json", "w") as outfile:
    json.dump(max_genes, outfile)

with open("Cerebellum_genes.json", "w") as outfile:
    json.dump(list(cpm_df.index), outfile)