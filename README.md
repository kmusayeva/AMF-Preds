### Soil Microbiome Prediction

This project addresses soil microbiome prediction problem in the context of multi-label classification. The focus here is specifically on arbuscular mycorrhizal fungi (AMF).
The AMF data used is publicly available on [Global AM Fungi](https://globalamfungi.com/). 

The project consists of two main directories:`soil_microbiome` and `m_lp`. 

`soil_microbiome/data/GlobalAMFungi/Species` contains the tabular data relating environmental variables with AMF abundancies. 
Currently, the provided data concern the taxonomic level of species.

`soil_microbiome/data_analysis` evaluates off-the-shelf multi-label classification methods, as well as appeals to label-propagation 
approaches from `m_lp` project. Due to the "power-law" like distribution of the species abundancies, only top frequent species should be selected (an option to be provided by a user). 
Furthermore, the label distribution should be kept similar across the folds to get a thorough evaluation via the family of F1-measures. 
In this project, this is done based on the stratified sampling method of Sechidis et al. [1].

The project `m_lp` implements label-propagation approaches (please check [2] for more information).

The entry point is `soil_microbiome/main.py`.


#### References
1. Sechidis, K., Tsoumakas, G., & Vlahavas, I. (2011). On the stratification of multi-label data. 
2. Musayeva, K., & Binois, M. (2023). Improved Multi-label Propagation for Small Data with Multi-objective Optimization. 
