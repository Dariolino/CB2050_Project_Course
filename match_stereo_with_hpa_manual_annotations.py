#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 2024

@author: Dario

This script is for matching 342 Purkinje cell-specific genes to manually 
annotated immunohistochemistry images from the Human Protein Atlas (HPA).
The annotations can have the following categories: High, Medium, Low, Not 
Detected, or No Data. These categories are assigned for general detection and
the subcellular locations (cytoplasm, dendrites, nucleus). 

In the last part of the script, part of the gene lists are exported as they
serve as the basis for making a UpSet Plot in a separate R script. 

The following file is required to run this script:
- purkinje_manual_annotations.csv: manually annotated IHC data for 342 Purkinje
  cell-specific genes
"""

#%% import libraries
import pandas as pd 
import matplotlib.pyplot as plt

#%%
# import manually annotated IHC data
df_manual_annotations = pd.read_csv("purkinje_manual_annotations.csv")

def plot_manual_annotations(df_annotations, location = "General"):
    """Make bar plots for the manually annotated IHC data

    Parameters:
    df_annotations (pandas dataframe): contains manually annotated IHC data
    location (str): protein location (General, Cytoplasm, Dendrites, Nucleus)

    Outputs:
    Bar plot showing how many genes are in each expression category
    """
    # HPA expression categories
    categories = ["High","Medium","Low","Not Detected","No Data"]

    vals = [len(df_annotations[df_annotations[location] == "High"]),    
            len(df_annotations[df_annotations[location] == "Medium"]),
            (df_annotations[df_annotations[location] == "Low"]),
            len(df_annotations[df_annotations[location] == "Not Detected"]),
            len(df_annotations[df_annotations[location] == "No Data"])]

    # bar colors
    colors = ["lightblue", "lightblue", "lightblue", "lightblue", "lightgrey"]

    # plot
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(categories, vals, color=colors, edgecolor="black")
    ax.bar_label(bars, fmt="%d", label_type="edge", padding=0.5, fontsize=12, 
                 fontweight="bold")
    ax.set_xlabel("Manually Annotated HPA Expression Categories",
                   fontsize=13.5)
    ax.set_ylabel("Number of Genes", fontsize=13.5)
    ax.set_title("Protein Presence of Purkinje Cell-Specific Genes - " 
                 + location, fontsize=13.5, fontweight="bold")
    ax.tick_params(axis="both", which="major", labelsize=12.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# make plot for general detection and for each subcellular location
plot_manual_annotations(df_manual_annotations, location = "General")
plot_manual_annotations(df_manual_annotations, location = "Cytoplasm")
plot_manual_annotations(df_manual_annotations, location = "Dendrites")
plot_manual_annotations(df_manual_annotations, location = "Nucleus")

#%% extract genes with high and medium expression for each subcellular location
# gene lists serve as a basis for a UpSet plot in a separate R script

# cytoplasm
genes_enr_in_cyto = set(
    df_manual_annotations.loc[
        (df_manual_annotations["Cytoplasm"] == "High") | 
        (df_manual_annotations["Cytoplasm"] == "Medium"), "gene_name"])

# dendrites
genes_enr_in_dendrites = set(
    df_manual_annotations.loc[
        (df_manual_annotations["Dendrites"] == "High") | 
        (df_manual_annotations["Dendrites"] == "Medium"), "gene_name"])

# nucleus
genes_enr_in_nucleus = set(
    df_manual_annotations.loc[
        (df_manual_annotations["Nucleus"] == "High") | 
        (df_manual_annotations["Nucleus"] == "Medium"), "gene_name"])

# export each set to a text file
with open("genes_enr_in_cyto.txt", "w") as f:
    f.write("\n".join(genes_enr_in_cyto))

with open("genes_enr_in_dendrites.txt", "w") as f:
    f.write("\n".join(genes_enr_in_dendrites))

with open("genes_enr_in_nucleus.txt", "w") as f:
    f.write("\n".join(genes_enr_in_nucleus))
