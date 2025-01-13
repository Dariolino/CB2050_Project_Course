 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 12:35:34 2023

@author: jan (& adjusted by dario)

This script is based on the original script from Jan Mulder (PhD). It is used
map snRNA-seq derived cell type-specific coexpression clusters onto the 
Stereo-seq spatial grid. The only adjustment to this script was changing the 
input data sets, the clusters to be analyzed, and theresolution value.
"""

#%% import necessary tools
import os 
import gc # for clearing unused objects
import shutil # for high level operations such as copying and moving files
import concurrent.futures # to parallelize tasks
import numpy as np
import pandas as pd
import tifffile as tiff # read and write tiff files (good for image data)
import matplotlib.pyplot as plt
from PIL import Image # Python Imaging Library, for opening, manipulating, and saving image files
from block_processor import process_block # process data in blocks
import fastparquet # for reading and writing parquet files
import math # for mathematical functions


#%%
# function to calculate pixel intensity values
def block_processing(df, r, spatial_sample, block_size, full_x, full_y, workers, multiprocess, inputvalue, mode):
    
    tmp_folder = 'output/02. Images/02.1. Raw prediction/' + spatial_sample + '/' + mode + 's/tmp_' + inputvalue  
    os.makedirs(tmp_folder, exist_ok=True)
    block_files = []

    # aave each block as a temporary file
    for x_start in range(0, full_x, block_size):
        x_end = min(x_start + block_size, full_x)
        for y_start in range(0, full_y, block_size):
            y_end = min(y_start + block_size, full_y)
            block_df = df[
                (df['x'] >= x_start - r) & (df['x'] < x_end + r) &
                (df['y'] >= y_start - r) & (df['y'] < y_end + r)
            ]
            block_file = os.path.join(tmp_folder, 'block_' + str(x_start) + '_' + str(y_start) + '.parquet')
            fastparquet.write(block_file, block_df)
            block_files.append(block_file)
    
    del df
    gc.collect()
    
    if multiprocess==False:
        # process each block sequentially
        result_files = []
        for block_file in block_files:
            print('Running ', block_file)
            result_file = process_block(block_file, r)
            result_files.append(result_file)
    
    
    if multiprocess==True:
        # process each block in a separate subprocess
        result_files = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(process_block, block_file, r) for block_file in block_files]
            for future in concurrent.futures.as_completed(futures):
                result_file = future.result()
                result_files.append(result_file)

    # concatenate all results into a single DataFrame
    result_dfs = [pd.read_parquet(result_file) for result_file in result_files]
    result_df = pd.concat(result_dfs, ignore_index=True)
    
    # clean up temporary files and folder
    del result_dfs
    shutil.rmtree(tmp_folder)
    gc.collect()
        
    return result_df

# function to make tiff image
def maketiff(df, r, spatial_sample, full_x, full_y, inputvalue, singlenuc_sample, cap, mode, exclude_gene):
    print("Create tiff image of prediction")
    new_image = np.zeros((full_x+1, full_y+1, 3), dtype=np.uint8)
    
    max_sum = df['block_sum_capped'].max()
    min_sum = df['block_sum_capped'].min()*-1

    normalized_values = df['block_sum_capped'].values
    x_values = df['x'].astype(int).values
    y_values = df['y'].astype(int).values

    # positive and negative values separation
    positive_mask = normalized_values >= 0
    negative_mask = ~positive_mask
    positive_values = normalized_values[positive_mask]
    negative_values = -normalized_values[negative_mask]

    # calculate values for green and blue channels
    new_image[x_values[positive_mask], y_values[positive_mask],  1] = (positive_values * (255 / max_sum)).astype(int)
    new_image[x_values[negative_mask], y_values[negative_mask], 2] = (negative_values * (255 / min_sum)).astype(int)

    # save or display the image
    if mode=='cluster': ### 3
        tiff.imsave('output/02. Images/02.1. Raw prediction/' + spatial_sample + '/clusters/cluster' + inputvalue + '_from_' + singlenuc_sample + '_mean_r' + str(r) + '_cap' + str(cap) + '.tif', new_image)
    if mode=='gene':
        if exclude_gene:
            tiff.imsave('output/02. Images/02.1. Raw prediction/' + spatial_sample + '/genes/' + inputvalue + '_from_' + singlenuc_sample + '_mean_r' + str(r) + '_cap' + str(cap) + '_exclude_' + inputvalue + '.tif', new_image)
        else:
            tiff.imsave('output/02. Images/02.1. Raw prediction/' + spatial_sample + '/genes/' + inputvalue + '_from_' + singlenuc_sample + '_mean_r' + str(r) + '_cap' + str(cap) + '.tif', new_image)


            
def real_exp_tiff(df, r, spatial_sample, full_x, full_y, inputvalue, singlenuc_sample, cap, mode):
    print("Create tiff image of real expression")
    new_image = np.zeros((full_x+1, full_y+1, 3), dtype=np.uint8)
    normalized_values = df['MIDCount'].values
    x_values = df['x'].astype(int).values
    y_values = df['y'].astype(int).values

    # calculate values for red channel
    new_image[x_values, y_values,  0] = (normalized_values * (255 / df['MIDCount'].max())).astype(int)

    # save or display the image
    if mode=='cluster':
        tiff.imsave('output/02. Images/02.1. Raw prediction/' + spatial_sample + '/clusters/cluster' + inputvalue + '_from_' + singlenuc_sample + '_mean_r' + str(r) + '_real_expr' + str(cap) + '.tif', new_image)
    if mode=='gene':
        tiff.imsave('output/02. Images/02.1. Raw prediction/' + spatial_sample + '/genes/' + inputvalue + '_from_' + singlenuc_sample + '_mean_r' + str(r) + '_real_expr' + str(cap) + '.tif', new_image)


# function to make png image
def makepng(df, r, spatial_sample, full_x, full_y, inputvalue, singlenuc_sample, cap, mode, exclude_gene):
    print("Create capped png image of prediction")
    new_image = Image.new("RGBA", (full_x+1, full_y+1), (255, 255, 255, 0)) # much faster with transparant background
    normalized_values = df['block_sum_capped'].values
    x_values = df['x'].astype(int).values
    y_values = df['y'].astype(int).values

    # positive and negative values separation
    positive_mask = normalized_values >= 0
    positive_values = normalized_values[positive_mask]
    positive_values_scaled = positive_values / positive_values.max()
    transparency_values = [max(100, int(255 * intensity)) for intensity in positive_values_scaled]
    pixels = new_image.load() 
    rgb_color = (255, 0, 0)
    for x, y, transparency in zip(x_values[positive_mask], y_values[positive_mask], transparency_values):
        # set RGB values with constant color and varying transparency
        pixels[x, y] = rgb_color + (transparency,)

    if mode=='cluster':
        new_image.save('output/02. Images/02.1. Raw prediction/' + spatial_sample + '/clusters/cluster' + inputvalue + '_from_' + singlenuc_sample + '_mean_r' + str(r) + '_cap' + str(cap) + '.png')
    if mode=='gene':
        if exclude_gene: 
            new_image.save('output/02. Images/02.1. Raw prediction/' + spatial_sample + '/genes/' + inputvalue + '_from_' + singlenuc_sample + '_mean_r' + str(r) + '_cap' + str(cap) + '_exclude_' + inputvalue + '.png')
        else:
            new_image.save('output/02. Images/02.1. Raw prediction/' + spatial_sample + '/genes/' + inputvalue + '_from_' + singlenuc_sample + '_mean_r' + str(r) + '_cap' + str(cap) + '.png')

    print("Create logged png image of prediction")
    new_image = Image.new("RGBA", (full_x+1, full_y+1), (255, 255, 255, 0)) # much faster with transparant background
    normalized_values = df['block_sum_log'].values
    x_values = df['x'].astype(int).values
    y_values = df['y'].astype(int).values

    # positive and negative values separation
    positive_mask = normalized_values >= 0
    positive_values = normalized_values[positive_mask]
    positive_values_scaled = positive_values / positive_values.max()
    transparency_values = [max(100, int(255 * intensity)) for intensity in positive_values_scaled]
    pixels = new_image.load() 
    rgb_color = (255, 0, 0)
    for x, y, transparency in zip(x_values[positive_mask], y_values[positive_mask], transparency_values):
        # set RGB values with constant color and varying transparency
        pixels[x, y] = rgb_color + (transparency,)

    if mode=='cluster':
        new_image.save('output/02. Images/02.1. Raw prediction/' + spatial_sample + '/clusters/cluster' + inputvalue + '_from_' + singlenuc_sample + '_mean_r' + str(r) + '_logged.png')
    if mode=='gene':
        if exclude_gene: 
            new_image.save('output/02. Images/02.1. Raw prediction/' + spatial_sample + '/genes/' + inputvalue + '_from_' + singlenuc_sample + '_mean_r' + str(r) + '_logged_exclude_' + inputvalue + '.png')
        else:
            new_image.save('output/02. Images/02.1. Raw prediction/' + spatial_sample + '/genes/' + inputvalue + '_from_' + singlenuc_sample + '_mean_r' + str(r) + '_logged.png')


            
def real_exp_png(df, r, spatial_sample, full_x, full_y, inputvalue, singlenuc_sample, cap, mode):
    print("Create png image of real expression")
    new_image = Image.new("RGBA", (full_x+1, full_y+1), (255, 255, 255, 0)) # much faster with transparant background
    normalized_values = df['MIDCount'].values
    x_values = df['x'].astype(int).values
    y_values = df['y'].astype(int).values

    # positive and negative values separation
    normalized_values_scaled = normalized_values / normalized_values.max() 
    transparency_values = [max(100, int(255 * intensity)) for intensity in normalized_values_scaled]
    pixels = new_image.load() 
    rgb_color = (0, 0, 255) # blue
    for x, y, transparency in zip(x_values, y_values, transparency_values):
        # Set RGB values with constant color and varying transparency
        pixels[x, y] = rgb_color + (transparency,)

    if mode=='cluster':
        new_image.save('output/02. Images/02.1. Raw prediction/' + spatial_sample + '/clusters/cluster' + inputvalue + '_from_' + singlenuc_sample + '_mean_r' + str(r) + '_real_exp' + str(cap) + '.png')
    if mode=='gene':
        new_image.save('output/02. Images/02.1. Raw prediction/' + spatial_sample + '/genes/' + inputvalue + '_from_' + singlenuc_sample + '_mean_r' + str(r) + '_real_exp' + str(cap) + '.png')



#%%
def make_images(singlenuc_sample, spatial_sample, mode, inputvalue, method, outputimageformat='tiff', r=10, full_x=None, full_y=None, multiprocess=False, workers=4, block_size=5000, cap=None, exclude_gene=False):
    
    ###########################################################################
    #                            Make output folders                          #
    ###########################################################################
    # make output folders
    os.makedirs('output/02. Images/02.1. Raw prediction/' + spatial_sample + '/genes/', exist_ok=True)
    os.makedirs('output/02. Images/02.1. Raw prediction/' + spatial_sample + '/clusters/', exist_ok=True)
    
    ###########################################################################
    #                              Load spatial data                          #
    ###########################################################################
    # Load the stereoseq CSV file (gem file) into a Pandas DataFrame
    print("Load spatial data")
    if method=='stereoseq':
        # CTR and FTD
        csv_file = 'input/' + spatial_sample + '.csv' 
        df = pd.read_csv(csv_file, engine='c')   
        # extend gemfile
        print("Expanding gemfile")
        df = df.rename(columns={'UMICount': 'MIDCount'}) 
        count_numbers = df['MIDCount'].unique() 
        df_keep = df[df['MIDCount'] == 1]
        for i in count_numbers[count_numbers > 1]: 
            #print(i)
            df_ext = df[df['MIDCount'] == i] 
            df_ext = pd.concat([df_ext] * i, ignore_index=True) 
            df_keep = pd.concat([df_keep, df_ext], ignore_index=True) 
        df = df_keep 
        del df_keep 
        del df_ext
        df['MIDCount'] = 1  # sets the counts to 1 in all rows --> that is okay since we have duplicated rows if count would be more than 1
        df = df.rename(columns={'bin_ID': 'bin1_ID'}) # renaming
        df['z'] = 0 # add a new coloumn z with value 0
    if method=='merfish':
        df = pd.read_csv('input/' + spatial_sample + '.barcodes.csv', engine='c')
        df['bin1_ID'] = df.index
        df = df.drop(columns=df.columns[[0,1, 5, 6]])
        df.columns = ['geneID', 'x', 'y', 'bin1_ID']
        df['MIDCount'] = 1
        df['x'] = df['x'] - df['x'].min()
        df['x_precise'] = df['x']
        df['x'] = df['x'].astype(int)#.values
        df['y'] = df['y'] - df['y'].min()
        df['y_precise'] = df['y']
        df['y'] = df['y'].astype(int)#.values
        df['z'] = 0
    if method=='xenium':
        csv_file = 'input/' + spatial_sample + '_transcripts.csv'
        df = pd.read_csv(csv_file, engine='c')
        df = df.drop(columns=df.columns[[1, 2, 7, 8,9]])
        df.columns = ['bin1_ID','geneID', 'x', 'y', 'z']
        df['MIDCount'] = 1   
        df['x_precise'] = df['x']
        df['y_precise'] = df['y']
        df['z_precise'] = df['z']
        df['x'] = df['x'].astype(int)#.values
        df['y'] = df['y'].astype(int)#.values
        df['z'] = df['z'].astype(int)#.values
        
        
    if exclude_gene is True and inputvalue not in df['geneID'].values: # check if we want to exclude some genes and if we do have the to-be-excluded gene in our data at all
        print(inputvalue + " was not measured so cannot be excluded")
        return
            
    
    ###########################################################################
    #                          Load single-nuc co-expression                  # input valueis either a gene or cluster string?
    ###########################################################################
    if mode=='cluster': # for cluster identification
        # load the single nuc z-score file (output from script 01) into a pandas dataframe
        print("Load z-scores for cluster " + inputvalue) # so inputvalue is a cluster?
        csv_file = 'output/01. Co-expression network/' + singlenuc_sample + '/Cluster' + inputvalue + '.csv'
        zf = pd.read_csv(csv_file, engine='c') 
        zf.columns = ['geneID', 'mean', 'sum'] # get the gene ID, mean and sum column
        
    if mode=='gene':
        # load the single nuc z-score file (output from script 01) into a Pandas DataFrame
        # check if gene is found in single cell data (do from the genes.csv file, because it takes long time to load the whole matrix)
        allgenes = pd.read_csv('output/01. Co-expression network/' + singlenuc_sample + '/coexpression_matrix_scaled_index.csv')
        allgenes = allgenes.rename(columns={"0": 'features'})
        if gene not in allgenes.features.values: # test = test.rename(columns={"0": 'features'})
            raise ValueError(f"'{gene}' not found in single cell data")
        # now load the large matrix    
        print("Load z-scores for gene " + inputvalue) 
        toskip = np.where(allgenes['features'] != gene)[0] #find row indices where the features column does not match the gene, creating an array of rows to skip --> allows for selective loading of only the necessary part of a large matrix.
        zf = pd.read_csv('output/01. Co-expression network/' + singlenuc_sample + '/coexpression_matrix_scaled.csv.gz', skiprows = toskip + 1, compression='gzip', engine='c')    
        zf = zf.T.tail(-1) # transposes zf and removes first (header row)
        zf['geneID'] = zf.index # adds a new column geneID that uses the current index.
        zf.columns = ['mean', 'geneID'] # renames columns so that the dataframe has two columns (mean and geneID)
        
    df=df.merge(zf, how='inner', on='geneID')  # only rows with matching geneID values in both dataframes are kept in the resulting df
    df = df[['x', 'y', 'z', 'bin1_ID', 'MIDCount', 'geneID', 'mean']] # mean z-score or a similar statistical value derived from zf

    del zf
    gc.collect()
    
    # fit the dimensions (of maximal rounded x, y and block size)
    if full_x is None:
        full_x=int(round(df['x'].max(), 0))
    if full_y is None:
        full_y=int(round(df['y'].max(), 0))
    if block_size is None:
        block_size=np.max([full_x, full_y])
    
    ###########################################################################
    #                          Identify NNC of each pixel                     # Nearest Neighbor Count/Classification # identification of the nearest neighboring pixel or set of pixels around a given pixel in an image or spatial grid
    ###########################################################################
    
    print("Running blocks...") # process the blocks (opionally with exclusing some genes)
    if exclude_gene and inputvalue in df['geneID'].values:  
        print("Excluding " + inputvalue + " in calculations...")
        df=block_processing(df[df['geneID'] != inputvalue], r, spatial_sample, block_size, full_x, full_y, workers, multiprocess, inputvalue, mode)
    else:
        df=block_processing(df, r, spatial_sample, block_size, full_x, full_y, workers, multiprocess, inputvalue, mode)
    
    # aggregate block_sum and MIDCount by coordinates
    print("Collapsing gemfile")
    df_tmp = pd.DataFrame(df.groupby(by=["x", "y", "z"], as_index=False)[["block_sum", "MIDCount"]].sum()) # group by x, y and z columns & aggregate block sum, and MID count (within each group) --> df_tmp holds the aggregated results
    df = df_tmp.merge(df[["bin1_ID", "x", "y", "z", "geneID"]], on = ['x','y', 'z'], how = "inner") # merges df_tmp with a subset of the original df that includes the columns bin1_ID, x, y, z, and geneID

    df = df.drop_duplicates()
    gc.collect()



    # plot histogram and export as pdf # examining the distribution of block sums
    plt.figure()
    fix, axs = plt.subplots(1, 1, figsize =(10,7), tight_layout = True)
    axs.hist(df['block_sum'], bins = 100)
    plt.axvline(x = 0, color = 'r')
    if mode=='cluster':
        plt.savefig('output/02. Images/02.1. Raw prediction/' + spatial_sample + '/clusters/cluster' + inputvalue + '_from_' + singlenuc_sample + '_mean_r' + str(r) + '_histogram.pdf', format = 'pdf')
    if mode=='gene':
        if exclude_gene: 
            plt.savefig('output/02. Images/02.1. Raw prediction/' + spatial_sample + '/genes/' + inputvalue + '_from_' + singlenuc_sample + '_mean_r' + str(r) + '_exclude_' + inputvalue + '_histogram.pdf', format = 'pdf')
        else:
            plt.savefig('output/02. Images/02.1. Raw prediction/' + spatial_sample + '/genes/' + inputvalue + '_from_' + singlenuc_sample + '_mean_r' + str(r) + '_histogram.pdf', format = 'pdf')

    #----------------#
    # plot histogram of logged values (with sign preservation)
    block_sums = df['block_sum']
    zero_mask = (block_sums == 0)
    negative_mask = (block_sums < 0)
    block_sums[zero_mask] = 1
    logged_block_sums = np.log(np.abs(block_sums))
    logged_block_sums[negative_mask] = -logged_block_sums[negative_mask]    
    logged_block_sums[zero_mask] = 0
    df['block_sum_log'] = logged_block_sums
        
    plt.figure()
    fix, axs = plt.subplots(1, 1, figsize =(10,7), tight_layout = True)
    axs.hist(df['block_sum_log'], bins = 100)
    plt.axvline(x = 0, color = 'r')
    plt.axvline(x = np.log(np.abs(cap)), color = 'g')
    
    if mode=='cluster':
        plt.savefig('output/02. Images/02.1. Raw prediction/' + spatial_sample + '/clusters/cluster' + inputvalue + '_from_' + singlenuc_sample + '_mean_r' + str(r) + '_histogram_log.pdf', format = 'pdf')
    if mode=='gene':
        if exclude_gene: 
            plt.savefig('output/02. Images/02.1. Raw prediction/' + spatial_sample + '/genes/' + inputvalue + '_from_' + singlenuc_sample + '_mean_r' + str(r) + '_exclude_' + inputvalue + '_histogram_log.pdf', format = 'pdf')
        else:
            plt.savefig('output/02. Images/02.1. Raw prediction/' + spatial_sample + '/genes/' + inputvalue + '_from_' + singlenuc_sample + '_mean_r' + str(r) + '_histogram_log.pdf', format = 'pdf')

    #----------------#

    # something weird here but shifting 1 pixel helps :-)
    df['x'] = df['x'] - 1
    df['y'] = df['y'] - 1
    
    # cap scale
    df['block_sum_capped'] =  df['block_sum'] # Capping values can be helpful for visualization purposes to prevent very high values (outliers) from skewing the scale of plots
    if cap is None:
        cap=math.ceil(df['block_sum'].max())
    df['block_sum_capped'][df['block_sum_capped'] > cap] = cap
    
    ###########################################################################
    #                              Plot real expression                       # plots real expression of the specified gene (diff formats available)
    ###########################################################################
    if mode=="gene":
        if inputvalue in df['geneID'].values:
            print("Plotting real expression")
            if outputimageformat=='both':  
                real_exp_tiff(df[df['geneID'] == inputvalue], r, spatial_sample, full_x, full_y, inputvalue, singlenuc_sample, cap, mode)
                real_exp_png(df[df['geneID'] == inputvalue], r, spatial_sample, full_x, full_y, inputvalue, singlenuc_sample, cap, mode)
            if outputimageformat=='tiff':
                real_exp_tiff(df[df['geneID'] == inputvalue], r, spatial_sample, full_x, full_y, inputvalue, singlenuc_sample, cap, mode)
            if outputimageformat=='png':  
                real_exp_png(df[df['geneID'] == inputvalue], r, spatial_sample, full_x, full_y, inputvalue, singlenuc_sample, cap, mode)
    

    ###########################################################################
    #                                Create images                            #
    ###########################################################################
    if outputimageformat=='both':  
        maketiff(df, r, spatial_sample, full_x, full_y, inputvalue, singlenuc_sample, cap, mode, exclude_gene)
        makepng(df, r, spatial_sample, full_x, full_y, inputvalue, singlenuc_sample, cap, mode, exclude_gene)
        
    if outputimageformat=='tiff':
        maketiff(df, r, spatial_sample, full_x, full_y, inputvalue, singlenuc_sample, cap, mode, exclude_gene)
        
    if outputimageformat=='png':  
        makepng(df, r, spatial_sample, full_x, full_y, inputvalue, singlenuc_sample, cap, mode, exclude_gene)
    
    ###########################################################################
    #                               Save outputfiles                          #
    ###########################################################################    
    print("Saving outputfiles")
    # save total counts under the mask
    df['MIDCount_blocksum_product'] = df['MIDCount'] * df['block_sum']
    df['MIDCount_totpixel_fraction'] = df['MIDCount'] / int(len(df))
    
    df_sub = df.loc[df['block_sum_capped'] > 0]  
    countsundermask = pd.DataFrame(df_sub.groupby(by=['geneID'])[['MIDCount', 'MIDCount_blocksum_product', 'MIDCount_totpixel_fraction']].sum())
    
    # export counts under the mask
    if mode=='cluster':
        countsundermask.to_csv('output/02. Images/02.1. Raw prediction/' + spatial_sample + '/clusters/cluster' + inputvalue + '_from_' + singlenuc_sample + '_mean_r' + str(r) + '_countsundermask.csv', index=True)
    if mode=='gene':
        if exclude_gene: 
            countsundermask.to_csv('output/02. Images/02.1. Raw prediction/' + spatial_sample + '/genes/' + inputvalue + '_from_' + singlenuc_sample + '_mean_r' + str(r) + '_exclude_' + inputvalue + '_countsundermask.csv', index=True)    
        else:
            countsundermask.to_csv('output/02. Images/02.1. Raw prediction/' + spatial_sample + '/genes/' + inputvalue + '_from_' + singlenuc_sample + '_mean_r' + str(r) + '_countsundermask.csv', index=True)    
            
    # save outputtable (includes negative )
    if mode=='cluster':
        df.to_csv('output/02. Images/02.1. Raw prediction/' + spatial_sample + '/clusters/cluster' + inputvalue + '_from_' + singlenuc_sample + '_mean_r' + str(r) + '_imagedataframe.csv.gz', index=False, compression='gzip')
    if mode=='gene':
        if exclude_gene: 
            df.to_csv('output/02. Images/02.1. Raw prediction/' + spatial_sample + '/genes/' + inputvalue + '_from_' + singlenuc_sample + '_mean_r' + str(r) + '_exclude_' + inputvalue + '_imagedataframe.csv.gz', index=False, compression='gzip')
        else:
            df.to_csv('output/02. Images/02.1. Raw prediction/' + spatial_sample + '/genes/' + inputvalue + '_from_' + singlenuc_sample + '_mean_r' + str(r) + '_imagedataframe.csv.gz', index=False, compression='gzip')


    
#%% run cluster predictions
#import time
#start_time = time.time()

#clusters=[0,4,5,11,13] # 0=Neurons, 4=Astrocytes,5=Oligodendrocytes,11=Microglia,13=Endothelial
#if __name__ == '__main__':
    #for cluster in clusters:
       # print("Running cluster " + str(cluster))
        #make_images(singlenuc_sample='CTR081_Fron',   # maybe datasetIB5219CB.h5ad --> but that is a h5ad file
         #           spatial_sample =  'H22.26.401.MTG.4000.expand.rep2',
          #          multiprocess = True,
           #         workers=15,
            #        block_size = 300,
             #       r=10, 
              #      cap=1000,
               #     #full_x = 21010-1, # width of the image, equals max x of the inputdata
                #    #full_y = 20458-1, # height of the image, equals max y of the inputdata
                 #   mode='cluster', # 'gene' or 'cluster'
                  #  inputvalue=str(cluster), # specify gene or cluster
                   # method='merfish', # 'stereoseq' 'merfish' 'xenium
                    #outputimageformat='both') # png or tiff or both. png has white background, tiff has black background
    
#end_time = time.time()    
#(end_time - start_time) / 60


#%% run gene predictions
#genes = ["GFAP","WDR49","NEFH","SPARC", "OPALIN","SLC17A7","GJA1", "GAD2", "CUX2","RELN", "RORB",  "ITGAM","SLC1A2","SOD1","ATP1B2"]
#genes = ["PFKFB2","SLC38A1", "CABLES1", "DPP10","CFAP47", "AEBP1","SLC1A3", "ZNF98","SLC4A4", "PLPP3", "CKB", "PAX6"]
#genes = ["SOX9", "CD44", "GRIA1", "S100B", "AQP4","ALDH1A1", "ALDH1L1", "SLC2A1","GRN", "C9orf72", "MAPT","TARDBP", "PDGFRB","COL1A1","DCN", "CLDN5", "FLT1", "MFSD2A", "GRM3", "GRM8", "ACTA2", "MYH11", "VWF", "EZR","GRIA1"]

# genes = ["GFAP"]
# genes = ["CA8", "HOMER3"]



# for gene in genes:
#     print("Running gene " + str(gene))
#     make_images(singlenuc_sample='datasetIB5219CB',
#                 spatial_sample = 'Combined_coordinate_file dario',
#                 multiprocess = False,
#                 workers=4,
#                 block_size = 2000,
#                 r=10, 
#                 cap=1000,
#                 #full_x = 21010-1, # width of the image, equals max x of the inputdata
#                 #full_y = 20458-1, # height of the image, equals max y of the inputdata
#                 mode='gene', # 'gene' or 'cluster'
#                 inputvalue=str(gene), # specify gene or cluster
#                 exclude_gene=False, # only relevant when mode='gene'
#                 outputimageformat='png', # png or tiff or both. png has white background, tiff has black background
#                 method='stereoseq' # 'stereoseq' 'merfish' 'xenium'
#                 )    
    
    

# csv_file = 'output/01. Co-expression network/' + singlenuc_sample + '/Cluster' + inputvalue + '.csv'
      #  zf = pd.read_csv(csv_file, engine='c') 
       # zf.columns = ['geneID', 'mean', 'sum'] # get the gene ID, mean and sum column

#single_nuc_sample = 
#csv_file_test = 'output/01. Co-expression network/datasetIB5219CB/Cluster3.csv'
#pd.read_csv('output/01. Co-expression network/datasetIB5219CB/Cluster3.csv', engine='c')

#csv_file = 'input/' + "Combined_coordinate_file dario" + '.csv' 
#pd.read_csv(csv_file, engine='c')   

##'/Users/dario/Desktop/mulder_project/Dario 2/input/Combined_coordinate_file dario.csv'


clusters = [0, 5, 8, 11, 12] # 0=Neurons, 5=Purkinje cells , 8=Astrocytes, 11=Microglia, 12=Oligodendrocytes
if __name__ == '__main__':
    for cluster in clusters:
        print("Running cluster " + str(cluster))
        make_images(singlenuc_sample='datasetIB5219CB',   # maybe datasetIB5219CB.h5ad --> but that is a h5ad file
                    spatial_sample =  'Combined_coordinate_file dario',
                    multiprocess = False,
                    workers=3,   
                    block_size = 300, #2000
                    r=20,  # increase to 20 or even higher
                    cap=1000,
                    #full_x = 21010-1, # width of the image, equals max x of the inputdata
                    #full_y = 20458-1, # height of the image, equals max y of the inputdata
                    mode='cluster', # 'gene' or 'cluster'
                    inputvalue=str(cluster), # specify gene or cluster
                    method='stereoseq', # 'stereoseq' 'merfish' 'xenium
                    outputimageformat='both') # png or tiff or both. png has white background, tiff has black background
# %%