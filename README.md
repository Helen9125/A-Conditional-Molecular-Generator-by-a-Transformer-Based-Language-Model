# A Conditional Molecular Generator by a Transformer-Based Language Model

## Abstract
A Conditional Molecular Generator by a Transformer-Based Language Model. Based on Transformer architecture and by employing relative position embedding and tie embedding, a chemical properties prediction model and a conditional molecular generation model are developed. The chemical properties prediction model first established is only built by encoder, which can predict the chemical properties value of a given structure based on its SMILES. In contrast, the conditional molecular generation model, built upon the trained chemical properties prediction model, includes decoder and can generate molecules with desired properties based on given backbone and chemical properties, showing well validity, uniqueness, and novelty. 
This study provides a new method and perspective for properties prediction and molecule generation, avoiding the tedious process and uncertainty of traditional molecule design. By implementing artificial intelligence, the proposed method provides information of certain structures in advance, reducing the time and cost required for drug design, and holds a significant value of drug discovery.

## Purpose
* Seeking to address the challenges by implementing a transformer-based generative model for drug discovery.
* By utilizing the capabilities such as predictions of these advanced neural networks, the project aims to expedite the process of designing molecules with therapeutic potential.

##Research Objectives
* Establishing a pre-trained model for properties prediction.
* Establishing a model for molecule generation based on the pre-trained model.
* Performing molecule generation tasks based on physiochemical properties.
* Evaluating the performance of the proposed model and comparing it with previous molecular generation models.
* Analyzing the advantages and limitations of transformer model on molecule generation tasks.

## Results
* The proposed prediction model can efficiently predict chemical properties of a large scale of molecules, and outperform traditional methods in speed while maintaining correlations above 0.7 for key properties.
![image](https://github.com/user-attachments/assets/b1a253cf-fb4a-4fe0-950e-c1f903a3660a)

* The molecule generation model performs well in generating molecules with desired properties, giving highly drug-like, easily synthesized molecules while maintaining over 80\% on performance metrics.
* The generation model is utilized to create analogs of TrkB receptor agonists, which proves the accessibility of the proposed model on conditionally generating real drugs.

## Methods
![image](https://github.com/user-attachments/assets/b671b6c7-f04a-4e17-a622-97db7b728288)

## Workflow of the Molecular Generator
![image](https://github.com/Helen9125/A-Conditional-Molecular-Generator-by-a-Transformer-Based-Language-Model/blob/main/graph/1.png)

## Datasets
Two datasets containing millions of molecules——MOSES and GuacaMol——are used to train the model.

## Architecture of Models
### Chemical properties prediction model
![image](https://github.com/Helen9125/A-Conditional-Molecular-Generator-by-a-Transformer-Based-Language-Model/blob/main/graph/2.png)
### Conditional molecular generation model
![image](https://github.com/Helen9125/A-Conditional-Molecular-Generator-by-a-Transformer-Based-Language-Model/blob/main/graph/3.png)

