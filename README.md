# A Conditional Molecular Generator by a Transformer-Based Language Model
## Abstract
A Conditional Molecular Generator by a Transformer-Based Language Model. Based on Transformer architecture and by employing relative position embedding and tie embedding, a chemical properties prediction model and a conditional molecular generation model are developed. The chemical properties prediction model first established is only built by encoder, which can predict the chemical properties value of a given structure based on its SMILES. In contrast, the conditional molecular generation model, built upon the trained chemical properties prediction model, includes decoder and can generate molecules with desired properties based on given backbone and chemical properties, showing well validity, uniqueness, and novelty. 
This study provides a new method and perspective for properties prediction and molecule generation, avoiding the tedious process and uncertainty of traditional molecule design. By implementing artificial intelligence, the proposed method provides information of certain structures in advance, reducing the time and cost required for drug design, and holds a significant value of drug discovery.
## Results
* The proposed prediction model can efficiently predict chemical properties of a large scale of molecules.
* The molecule generation model performs well on generating molecules with desired properties.
* The generation model is utilized to create analogs of TrkB receptor agonist, which proves the accessibility of the proposed model on conditionally
## Workflow of the Molecular Generator
![image](https://github.com/Helen9125/A-Conditional-Molecular-Generator-by-a-Transformer-Based-Language-Model/blob/main/graph/1.png)
## Datasets
Two datasets containing millions of molecules——MOSES and GuacaMol——are used to train the model.
 generating real drugs. 
## Architecture of Models
### Chemical properties prediction model
![image](https://github.com/Helen9125/A-Conditional-Molecular-Generator-by-a-Transformer-Based-Language-Model/blob/main/graph/2.png)
### Conditional molecular generation model
![image](https://github.com/Helen9125/A-Conditional-Molecular-Generator-by-a-Transformer-Based-Language-Model/blob/main/graph/3.png)

