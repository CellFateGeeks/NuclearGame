[![DOI](https://zenodo.org/badge/348290747.svg)](https://zenodo.org/badge/latestdoi/348290747)

# NuclearGame
NuclearGame (NG) is a Python module for segmenting cellular nuclei, measuring nuclear features, and predicting cell fate (based on these features).

NG is part of the project: *Cell nucleus-based imaging analysis with machine learning to predict cell reprogramming into neurons*. For this reason, NG was developed mainly for analysing nuclear changes throughout the process of glia-to-neuron reprogramming. 

Although NG was developed for studying nuclei during reprogramming, **any cell culture image with staining of the nuclei can be segmented, analysed, and clustered, with this programme.**

However, if NG will be used for a cell reprogramming analysis, we recommend that the cell culture images contain a protein reporter showing cell transduction and a  marker showing the final state of the reprogramming, besides the staining of the nuclei. In our case, we employed:
- Nuclear staining : DAPI (specifically, DNA staining)
- Transduction reporter : RFP (Red Fluorescent Protein)
- Final state marker : Tuj1 (neuronal marker)
- Extra marker : GFAP (glial cell marker)

## Input Files
- **Epifluorescence microscopy images** : Cell culture images (currently, only CZI files are supported) containing, at least, one channel with the staining of the nuclei. 

## Output Files
- **output.json.gz** : Compressed JSON file containing all data and metadata of the images, nuclear segmentation, and raw information of the measured nuclear features.
- **raw_output.csv** : CSV file containing raw information of the measured nuclear features.
- **filtered_output.csv** : CSV file contaning filtered information of the measured nuclear features.

## Usage
Examples of usage for segmentation, analysis, and clustering can be found in **/Notebooks/** as **ng_segmentation.ipynb**, **ng_analysis.ipynb**, and **ng_clustering.ipynb**, respectively.

## Citation
Gabriel Emilio Herrera-Oropeza, & Marcelo J Salierno. (2021, March 22). CellFateGeeks/NuclearGame: Pre-release of NuclearGame (Version v0.1.1). Zenodo. http://doi.org/10.5281/zenodo.4626447
