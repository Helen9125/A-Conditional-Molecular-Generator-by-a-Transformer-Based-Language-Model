# A Conditional Molecular Generator by a Transformer-Based Language Model
## Abstract
A Conditional Molecular Generator by a Transformer-Based Language Model. Based on Transformer architecture and by employing relative position embedding and tie embedding, a chemical properties prediction model and a conditional molecular generation model are developed. The chemical properties prediction model first established is only built by encoder, which can predict the chemical properties value of a given structure based on its SMILES. In contrast, the conditional molecular generation model, built upon the trained chemical properties prediction model, includes decoder and can generate molecules with desired properties based on given backbone and chemical properties, showing well validity, uniqueness, and novelty. 
This study provides a new method and perspective for properties prediction and molecule generation, avoiding the tedious process and uncertainty of traditional molecule design. By implementing artificial intelligence, the proposed method provides information of certain structures in advance, reducing the time and cost required for drug design, and holds a significant value of drug discovery.
## Results
* The proposed prediction model can efficiently predict chemical properties of a large scale of molecules.
* The molecule generation model performs well on generating molecules with desired properties.
* * The generation model is utilized to create analogs of TrkB receptor agonist, which proves the accessibility of the proposed model on conditionally
## Workflow of the Molecular Generator
![image](https://github.com/user-attachments/assets/e8af4e2c-c8a8-4b49-87eb-1544dca4ae2c)
## Datasets
Two datasets containing millions of molecules——MOSES and GuacaMol——are used to train the model.
 generating real drugs. 
## Architecture of Models
### Chemical properties prediction model
![image](https://github.com/user-attachments/assets/19729a9b-3ce9-41b6-95c9-0a137caf3784)
### Conditional molecular generation model
![image](https://github.com/user-attachments/assets/7e0c3561-d6bc-45fc-b54e-649470a8925e)

