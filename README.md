### Soil Microbiome Prediction

This project addresses soil microbiome prediction problem in the context of multi-label classification. The focus here is specifically on arbuscular mycorrhizal fungi (AMF).
The AMF data used is publicly available on [Global AM Fungi](https://globalamfungi.com/). 

#### Project structure
The project consists of two main directories:`soil_microbiome` and `m_lp`. 

`soil_microbiome/data/GlobalAMFungi/Species` contains the tabular data relating environmental variables with AMF abundancies. 
Currently, the provided data concern the taxonomic level of species.

`soil_microbiome/data_analysis` evaluates off-the-shelf multi-label classification methods, as well as appeals to label-propagation 
approaches from `m_lp` project. Due to the "power-law" like distribution of the species data, only top frequent species are selected (an option to be provided by a user). 
Furthermore, the label distribution is kept similar across the folds based on stratified sampling method of 
Sechidis et al. (2011), On the stratification of multi-label data.

`m_lp` implements label-propagation approaches. For more information on multi-label 
propagation, please see Musayeva, K., & Binois, M. (2023). Improved Multi-label Propagation for Small Data with Multi-objective Optimization. 

The entry point is `soil_microbiome/main.py`.
