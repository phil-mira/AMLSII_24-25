# AMLSII_24-25

This repo contains the models and training pipeline for the 2020 SIIM-ISIC Melanoma Classification challenge. To run the models in the repo the data from the competition Kaggle page needs to be downloaded to a folder called data with the given name of the download remaining the same as siim-isic-melanoma-classification, https://www.kaggle.com/competitions/siim-isic-melanoma-classification/overview.

Additionally, there is an additional file containing all the duplicate images that needs to be downloaded in order to run the code. https://www.kaggle.com/competitions/siim-isic-melanoma-classification/discussion/161943

With the relevant data added the main training pipeline can be run using main.py. This will create the models and train them saving the weights to the saved_models directory. To view the analysis of the results use the results.ipynb notebook to carry out the validation of the data and compute the relevant figures. Additionally, to view the plots mentioned during the data description section of the report see the exploration.ipynb notebook. 
