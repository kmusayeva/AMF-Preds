### Soil Microbiome Prediction

This project addresses soil microbiome prediction problem in the context of multi-label classification. The focus here is specifically on arbuscular mycorrhizal fungi (AMF).
The AMF data used is publicly available on [Global AM Fungi](https://globalamfungi.com/). 

The project consists of two directories:`soil_microbiome` and `m_lp`.

`soil_microbiome/data/GlobalAMFungi` contains the tabular data relating environmental variables with species abundancies.

`soil_microbiome/data_analysis` implements off-the-shelf classification methods as well as appeals to label-propagation 
approaches. 

`m_lp` implements label-propagation approaches. For more information on multi-label 
propagation, please see Musayeva, K., & Binois, M. (2023). Improved Multi-label Propagation for Small Data with Multi-objective Optimization. 

