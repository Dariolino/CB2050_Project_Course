#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: saga & dario

This script is based on a script from Saga Bolund (M.Sc.) which visualizes
transcripts on the Stereo-seq spatial grid. The script was adjusted to 
generate combined images for stereo-seq data. The images are created by 
assigning a unique color to each gene in a list of 3 genes. The genes are then
visualized in the same image.

In the second part of the script, the script is used to generate images for a
list of 364 genes from the Human Protein Atlas (HPA) that are specific to
Purkinje cells. The images were generated and manually inspected to identify
Purkinje cell markers in the stereo-seq data.
"""

#%%
# import modules
import pandas as pd # for using dataframe
import numpy as np #  for using arrays
import matplotlib.pyplot as plt # for saving the image in the end

#%%
# auxilliary functions

def check_list_of_3_genes(list_of_3_genes, input_df):
    """Ensures that input list contains 3 genes, which are present in input_df
    
    Parameters: 
    list_of_3_genes (list): contains 3 gene names as strings
    input_df (pandas df): contains "geneID" column with gene names
    
    Returns:
    - if list contains less than 3 elements: raises ValueError 
    - if gene is not present: prints missing gene name
    """
    # ensure the list has 3 genes
    if len(list_of_3_genes) != 3:
        raise ValueError("The input must be a list of exactly 3 genes to use", 
                         " RGB color mapping.")
    
    # check if each gene in list_of_3_genes is in the geneID column
    missing_genes = []
    for gene in list_of_3_genes:
        if gene not in input_df['geneID'].values:
            missing_genes.append(gene)
    if missing_genes:
        print("The following genes were not found in the geneID column:")
        for gene in missing_genes:
            print(gene)
            

def assign_RGB_color_to_3_genes(list_of_3_genes):
    """Creates dictionary with gene name as key and RGB code as value

    Parameters:
    list_of_3_genes : contains 3 gene names as strings

    Returns:
    dictionary with gene names as keys, and as values the assigned RGB code:
        - red, green and blue RGB code for 1st, 2nd and 3rd gene respectively
    """
    return {list_of_3_genes[0]: (255, 0, 0),  # red
            list_of_3_genes[1]: (0, 255, 0),  # green
            list_of_3_genes[2]: (0, 0, 255)}   # blue


def get_max_dims_based_on_resolution(input_df, resfactor):
    """Scales the maximum x & y value in the input dataframe to the resolution
    
    Parameters:
    input_df (pandas df): contains "geneID" column with gene names
    resfactor (int) : resolution factor for scaling

    Returns:
    max_x (int): scaled maximum x coordinate
    max_y (int): scalled maximum y coordiante
    """
    max_x = (input_df['x'].max() // resfactor) + 2
    max_y = (input_df['y'].max() // resfactor) + 2
    return max_x, max_y


def subset_and_scale_gene_with_resfactor(input_df, gene_name, resfactor):
    """Subset your dataframe to your gene and scale it based on the resolution"
    
    Parameters:
    input_df (pandas df): contains geneID column, x, y and MIDCount values
    gene_name (string): name of the gene
    resfactor : resolution factor for scaling

    Returns:
    x_values (numpy.ndarray): scaled x value for the gene after grouping and scaling
    y_values (numpy.ndarray): scaled y value for the gene after grouping and scaling
    """
    # subset the dataframe for the gene and adjust coordinates for resolution
    df_sub = input_df[input_df['geneID'] == gene_name]
    df_sub.loc[:, 'x'] = (df_sub['x'] / resfactor).round().astype(int)
    df_sub.loc[:, 'y'] = (df_sub['y'] / resfactor).round().astype(int)
    
    # group by x, y and sum the counts
    grouped = df_sub.groupby(['x', 'y'])['MIDCount'].sum().reset_index()
    x_values = grouped['x'].values
    y_values = grouped['y'].values
    
    return x_values, y_values


def get_color_definitions():
    """Provides the RGB codes for a set of colors
    
    Returns:
    A set of np.arrays for each RGB code of the colors
    """
    dark_yellow = np.array([139, 139, 0], dtype=np.uint8)
    dark_magenta = np.array([139, 0, 139], dtype=np.uint8)
    dark_cyan = np.array([0, 139, 139], dtype=np.uint8)
    black = np.array([0, 0, 0], dtype=np.uint8)
    white = np.array([255, 255, 255], dtype=np.uint8)
    
    return dark_yellow, dark_magenta, dark_cyan, black, white


def assign_correct_color(pixel_col, i, color, white, red, green, dark_yellow,
                         dark_magenta, dark_cyan, black):
    """Custom function to return a specified color based on current pixel 
    color and the iteration step for the gene processing (3 genes in total)
    
    Parameters:
    pixel_col (np.array): contains a given RGB code 
    i (int): index within a for loop iterating through 3 genes:
                            - corresponding to the colors red, green and blue
    white, red, green, dark_yellow, dark_magenta, dark_cyan, black (np.array):
        predefined colors 
    
    Returns:
    updated color (np.array)
    """
    if np.array_equal(pixel_col, white):
        return color # if background white --> assign predefined gene color
    elif np.array_equal(pixel_col, red): 
        if i == 1:
            return dark_yellow  # red + green --> assign dark yellow
        elif i == 2:
            return dark_magenta  # red + blue --> assign dark magenta
    elif np.array_equal(pixel_col, green): 
        if i == 2:
            return dark_cyan  # green + blue -> assign dark cyan
    elif np.array_equal(pixel_col, dark_yellow) or np.array_equal(pixel_col, dark_magenta) or np.array_equal(pixel_col, dark_cyan):
        return black # if all three colors overlap -> assign black
    

# main function

def make_combined_image(spatial_sample, input_df, list_of_3_genes, resfactor=10,
                        blend_on_black=False):
    """Creates a combined png file for three genes in the stereo-seq data
    
    Parameters:
    spatial_sample (str): used for output file name
    input_df (pandas df): contains geneID column, x, y and MIDCount values
    list_of_3_genes (list): contains 3 gene names as strings
    resfactor (int) : resolution factor for scaling
    blend_on_black (bool): uses a black background if changed to True
    
    Saves:
    png image to the harcoded output path (can be changed if desired)
    """
    check_list_of_3_genes(list_of_3_genes, input_df)
    
    # assign RGB color to each gene (red, green, blue)
    gene_colors = assign_RGB_color_to_3_genes(list_of_3_genes)
    red, green, blue = gene_colors.values()
    
    # calculate maximum dimensions based on resfactor and the input dataframe
    max_x, max_y = get_max_dims_based_on_resolution(input_df, resfactor)

    # initialize image with white background (or black if blend_on_black is True)
    new_image = np.ones((max_y, max_x, 3), dtype=np.uint8) * (0 if blend_on_black else 255)
    
    # get color definitions for assigning the correct color for the pixel
    dark_yellow, dark_magenta, dark_cyan, black, white = get_color_definitions()

    # loop over each gene and its assigned color (i counts the step of the gene processing)
    for i, (gene, color) in enumerate(gene_colors.items()):
        x_vals, y_vals = subset_and_scale_gene_with_resfactor(input_df, gene, resfactor)
        
        for x, y in zip(x_vals, y_vals):  # iterate through the pairs of coordinates
            pixel_color = new_image[y, x, :].copy() # snapshot of the pixel color  
            if blend_on_black:
                # additive blending on black
                new_image[y, x, :] = np.minimum(pixel_color + color, 255)
            else:
                # define color of the pixel based on previous color of the pixel
                new_image[y, x, :] = assign_correct_color(pixel_color, i,
                                                          color, white,red,
                                                          green, dark_yellow,
                                                          dark_magenta,
                                                          dark_cyan, black)

    # save the image
    output_file = f'/Users/dario/Downloads/{spatial_sample}_RGB_genes_{list_of_3_genes[0]}_{list_of_3_genes[1]}_{list_of_3_genes[2]}_res{resfactor}.png'
    plt.imsave(output_file, new_image)


#%%
# apply functions

# load spatial data
csv_file = "/Users/dario/Desktop/mulder_project/Task_2/Cell_segmentation_scripts/input/Combined_coordinate_file dario.csv"
df = pd.read_csv(csv_file) 

# example gene lists
genes_1 =["ALDOC", "GFAP", "CALB1"]#"ALDH1A1","NEFL", "MAP2", "CALY", "BMP4", "IGF1", "CXCL8", "MPO"]#"SLC18A2", "SNAP25", "KCNJ6", "UCHL1", "SYT1", "TUBB3", "YWHAG", "THY1", "NEFL", "ANK1", "ANXA6", "CLU", "SPARCL1", "MOG", "MBP", "APOE", "SERPINA", "VIM", "OLIG2",  "ALDH1A1"]#"AQP2","AQP1","SLC6A3","SNCA", "CDNF","AQP3", "AQP4", "AQP5", "AQP6", "AQP7", "AQP7B", "AQP8", "AQP9", "AQP10", "AQP11",

       #"AQP12A", "AQP12B","GFAP", "WDR49", "AQP4","OPALIN","SLC17A7","GJA1", "GAD2", "CUX2", "RELN", "PLP1", "ITGAM", "NEFM",

       #'TH', "SLC6A2", "DRD3" ,"DRD4" ,"DRD5", "CKB", "CARTPT", "CCK", "TAC1", "GAL", "ALDH1L1", "NPY"] #,"SLC6A2", "DRD3","DRD4", "DRD5", "CDNF", "SNCA", "SLC6A3", "AQP1", "AQP2","AQP3", "AQP4", "AQP5", "AQP6", "AQP7", "AQP7B", "AQP8", "AQP9", "AQP10", "AQP11",

       #"AQP12A", "AQP12B","GFAP", "WDR49", "AQP4","OPALIN","SLC17A7","GJA1", "GAD2", "CUX2", "RELN", "PLP1", "ITGAM", "NEFM", "PLP1"] #STMN2", "MT-TQ", "MT-TG"
       
genes_2 = ["AQP4","OPALIN","SLC17A7"]
genes_11 = ["PVALB", "PCP4", "CALB1"]
genes_12 = ["PCP4", "PVALB","CALB1"]
genes_13 = ["HOMER3", "ITPR1", "ITM2B"]
genes_14 = ["GRID2", "PCP2", "YWHAH"]   

astrocyte_subset_1 = ["S100B", "ALDH1L1", "AQP4"]
astrocyte_subset_2 = ["SLC1A2", "", "SOX9"]

microglia_subset_1 = ["P2RY12", "CX3CR1", "ITGAM"]
microglia_subset_2 = ["PTPRC", "CX3CR1", "ITGAM"]

oligodendrocyte_subset_1 = ["MBP", "MOG", "PLP1"]
oligodendrocyte_subset_2 = ["OLIG2", "SOX10", "CNP"]
oligodendrocyte_subset_3 = ["MAG", "SOX10", "CNP"]

purkinje_genes = ["PPP1R17","PVALB", "HOMER3", "CA8", "PCP4", "CALB1", "ITPR1",
                   "SLC1A6"]
astrocyte_genes = ["S100B", "ALDH1L1", "AQP4", "SLC1A2", "SOX9"] # astrocyte markers # Bergmann glia has SLC1A3 and SOX2
microglia_genes = ["P2RY12", "CX3CR1", "ITGAM", "PTPRC"] # cluster 11
oligodendrocyte_genes = ["MBP", "MOG", "PLP1", "OLIG2", "SOX10", "CNP", "MAG"]


# example usage
# iterate through genes_12 until genes_14 and name the spatial sample input with the same number
for i in range(12, 15):
    make_combined_image(spatial_sample = f"Combined_{i}", 
                        input_df = df,
                        list_of_3_genes = eval(f"genes_{i}"),
                        resfactor = 10,
                        blend_on_black=False)



#%% 
###############################################################################
###############################################################################
###############################################################################

# Analysis of the list of 364 genes from Evelina
# https://www.proteinatlas.org/search/normal_expression:Cerebellum;Purkinje+cells+-+cytoplasm%2Fmembrane;Medium%2CHigh)

HPA_purkinje_genes_df = pd.read_csv('/Users/dario/Downloads/normal_expression_Cerebellum_Purkinje.tsv',
                                    sep='\t')

# extract the gene column as a list of strings
HPA_purkinje_gene_list = HPA_purkinje_genes_df["Gene"].tolist()

# get a list with the intersection of the HPA data and the stereo-seq genes
def create_matching_gene_list(input_df, HPA_list):
    matching_genes = list(set(HPA_list).intersection(input_df['geneID'].unique()))
    return matching_genes
    
matched_genes = create_matching_gene_list(df, HPA_purkinje_gene_list) # all 364 genes matched 


# this function is used to generate the stereo-seq images for all matched genes
# a batch number of 3 was used (3 genes are always put into one image in different colors)
def extract_HPA_genes_in_batches_and_save_images(long_gene_list, batch_size, stereo_df, resfactor = 20):
    iteration_count = 0 # to allow for printing the iteration step
    
    # loop over the gene list in increments of the batch_size
    for i in range(0, len(long_gene_list), batch_size):
        # create a subset of the current batch
        subset_list_of_3_genes = long_gene_list[i:i + batch_size]
        
        # generate a image for each subset
        make_combined_image(spatial_sample = "Combined", 
                            input_df = stereo_df,
                            list_of_3_genes = subset_list_of_3_genes,
                            resfactor = resfactor)
        
        print(f"Iteration {iteration_count}: Image for genes {subset_list_of_3_genes} was created.")
        iteration_count += 1

# generate images for all genes using a resolution of 20 
# manually inspect the images and look for Purkinje cell markers
extract_HPA_genes_in_batches_and_save_images(matched_genes, 3, df, 20)

# potentially do last three genes manually. Here: technically not required.
genes_last = matched_genes[-4:-1] # ["FAM81A", "PJA2", "VPS37B"]

make_combined_image(spatial_sample = "Combined",
                    input_df = df,
                    list_of_3_genes = genes_last,
                    resfactor = 20,
                    blend_on_black=False)

# Results for the matched gene list:
# Best markers in stereo-seq data: PPP1R17,PVALB, HOMER3, CA8, NPTX1, PCP4, CALB1, ITPR1, SLC1A6
# Comparison to antibody staining from HPA: all match well with protein expression