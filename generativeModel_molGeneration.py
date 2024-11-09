'''
Written by Jinhoon Jeong (Dong-A ST), as a participant of the LAIDD mentoring project
All scripts in bs_denovo were provided as part of the mentoring project and include some minor modifications.
This script is also mostly referred to the original script provided from the project.
'''
import torch
import pandas as pd
import numpy as np
import math, os
from bs_denovo import bs_chem
from bs_denovo.lang_vae import LangVAE
from bs_denovo.vae_lstm import LSTMEncoder, LSTMDecoder
from bs_denovo.vocab import SmilesVocabulary

dev='cuda'
os.system('mkdir -p postproc')

ckpt_e200 = torch.load("./models/vae_e200.ckpt", weights_only=False, map_location=dev)
vae_inst = LangVAE.construct_by_ckpt_dict(ckpt_e200,
                    VocClass=SmilesVocabulary, EncClass=LSTMEncoder,
                    DecClass=LSTMDecoder, dev=dev)

# random generation
rand_gens, _ = vae_inst.randn_samples(50000, bsz=100, greedy=False)
# create a Dataframe for random generation, and save as csv
rg_df = pd.DataFrame(rand_gens, columns=['gen'])
rg_df.to_csv("./postproc/random_gen.csv", index=False)

# guided generation
# checking SGK1 binding affinity data
ba_df = pd.read_csv("./data/BindingDB_SGK1_IC50_duplicatesRemoved.tsv", sep='\t')  # binding affinity dataframe
ba_df['pIC50'] = ba_df['IC50'].apply(lambda x: -math.log10(x*1E-9))

# retrieve known SGK1-active compounds
cutoff = 7
activ_df = ba_df[ba_df["pIC50"] >= cutoff]
print(f'# of mols whose pIC50 > 7: {activ_df.shape[0]}')
known_actives = activ_df['canonical_smiles'].tolist()
known_actives_lbls = activ_df['pIC50'].tolist()

# dataset/dataloader of known actives
from bs_denovo.lang_data import StringDataset
from torch.utils.data import DataLoader

kn_act_ds = StringDataset(vae_inst.decoder.voc, known_actives)  # kn_act : known actives
kn_act_dl = DataLoader(kn_act_ds, batch_size=100, shuffle=False, collate_fn=kn_act_ds.collate_fn)

# collecting mu and logvar of known actives
mu_col, logvar_col = [], []
for bat_seqs in kn_act_dl:
  _, mu, logvar, _ = vae_inst.encoder.forward_by_tgt(bat_seqs)
  mu_col.append(mu.detach().cpu())
  logvar_col.append(logvar.detach().cpu())
# stacking the mu/logvar in a single tensor
kn_mus = torch.cat(mu_col, dim=0)
kn_logvars = torch.cat(logvar_col, dim=0)


def zvec_sampling(mu, logvar, bsz):
  d_latent = mu.shape[0]  # assume mu,logvar are single vectors
  std = torch.exp(0.5*logvar)
  eps = torch.randn((bsz, d_latent))
  zvec = mu + eps*std
  return zvec

  # guided generation - 25 samples per guiding molecule
# create list of records {"guide_smi":..., "known_pIC50":..., "zvec":...}
sam_per_guide = 80
records = []
for guide_smi, known_pIC50, mu, logvar in zip(known_actives, known_actives_lbls, kn_mus, kn_logvars):
  zvec = zvec_sampling(mu, logvar, sam_per_guide)
  for i in range(sam_per_guide):
    records.append({"guide_smi":guide_smi, "known_pIC50":known_pIC50, "zvec":zvec[i].tolist()})

# build a Dataframe using the records
gg_df = pd.DataFrame.from_records(records)

# decode each of zvec
from bs_denovo.lang_data import VectorDataset
zvecs = gg_df['zvec'].tolist()
zvec_ds = VectorDataset(zvecs)
zvec_dl = DataLoader(zvec_ds, batch_size=100, shuffle=False, collate_fn=zvec_ds.collate_fn)

gens = []  # collecting generations
for bat_z in zvec_dl:
  smis, _ = vae_inst.decoder.decode2string(bat_z, greedy=True, max_len=250)
  gens.extend(smis)

# add generation column and save to csv
gg_df['gen'] = gens
print(f'Total # of random molecules: {rg_df.shape[0]}')
print(f'Total # of guided molecules: {len(gens)}')
gg_df.to_csv("./postproc/guided_gen.csv", index=False)
