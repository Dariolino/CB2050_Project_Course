# CB2050_Project_Course

This repository contains all the scripts used to generate the data for the final report for the KTH course CB2050 Project in molecular life science. Within all the scripts, a description of their usage/purpose together with a concrete authorship of the scripts can be found, since most scripts used in this project were provided by different people in the research group of Jan Mulder (PhD).
To generate the results from the final report, the following scripts were run in the following order:
1.	make_co_expression_network.py – for snRNA-seq processing and deriving cell type-specific co-expression sets
2.	make_stereo_seq_prediction.py – predict cell type-specific co-expression sets in Stereo-seq data
3.	clean_stereo_seq_prediction.py – refine the prediction of the previous script
4.	combine_stereo_seq_cluster_images.py – merge previous cell cluster predictions into one single image
5.	perform_pseudobulking.py – pseudobulk counts of the predicted cell type coordinates
6.	pseudobulk_analysis_plots.py – derive Purkinje cell-enriched genes and match them to cerebellum-specific genes (as annotated in the Human Protein Atlas)
7.	combined_stereo_seq_image_generator.py – visualize Stereo-seq transcripts on the Stereo-seq spatial grid
8.	enrichment_analyis_purkinje_cells.qmd – GO term and KEGG pathway enrichment analysis of Purkinje cell-enriched genes
9.	match_stereo_with_hpa_manual_annotations.py – visualize the protein presence and subcellular location of the derived Purkinje-cell enriched genes based on manually annotated immunohistochemistry images from the Human Protein Atlas
10.	upset_plot_for_hpa_manual_annotations.qmd – create UpSet plot to see overlaps in the subcellular location of Purkinje-cell specific proteins detected in the Human Protein Atlas
