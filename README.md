All scripts in "bs_denovo" directory were provided as part of the mentoring project and include some minor modifications.

The scripts below are also based on what we learned in class and reference the original code.

1. predictiveModel_train_with_ML_models.py
- for training RF, XGBoost, GBM, SVM models with SGK1 data.
2. generativeModel_train_with_VAE.py
- for training VAE model with kinase inhibitor library data.
3. generativeModel_molGeneration.py
- From the trained VAE model, this generates molecules.
4. generativeModel_evaluation_models.py
- for evaluating VAE models
5. filtering.py
- for filtering molecules as virtual screening
6. attentionModel_train_allKinaseLibrary.py
- for training attention-based model (lig: GCN, prot: 1dCNN) with kinase-ligand activity data
7. SGK1_finetuning.py
- for finetuning of attention-based model with SGK1 data 

Special thanks to Professor Nam's group in GIST for their dedicated guidance and invaluable support. 
