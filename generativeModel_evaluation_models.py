'''
Written by Jinhoon Jeong (Dong-A ST), as a participant of the LAIDD mentoring project
All scripts in bs_denovo were provided as part of the mentoring project and include some minor modifications.
'''
import matplotlib.pyplot as plt
import os, re
from bs_denovo.lang_vae import LangVAE
from bs_denovo.vocab import SmilesVocabulary
from bs_denovo.vae_lstm import LSTMEncoder, LSTMDecoder
from bs_denovo.lang_data import StringDataset
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import torch, re
from bs_denovo.gen_eval import standard_metrics
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# variables
base_dir1 = '/home/dongaST/Jinhoon/LAIDD_project'
base_dir = './'
dev = 'cuda'

# tokens
tokens = ['H', 'B', 'C', 'N', 'O', 'F', 'Na', 'Mg', 'Al', 'Si','P', 'S', 'Cl', 'Zn', 'As', 'Se', 'Br', 'Sr', 'Ag', 'Sn','Te', 'I', 'V', 'Li', 'Cr',
		  'c', 'n', 'o', 'b', 's', 'p', 'se', 'te', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0',
		  '-', '=', '#', '$', ':', '/', '\\', '+', '%', '[', ']', '(', ')', '@', '.',
		  '<CLS>', '<BOS>', '<EOS>', '<PAD>', '<MSK>', '<UNK>']


# vocabulary object (function: tokens, tok2id, id2tok, vocab_size, encode, decode, tokenize)
smiles_voc = SmilesVocabulary(list_tokens=tokens)

# load data
df = pd.read_csv(f'{base_dir1}/training_data/splitted_data_canonical_removed.tsv', sep='\t')
smiles_trn = df.loc[df['split']=='train', 'smiles'].tolist()
smiles_vld = df.loc[df['split']=='validation', 'smiles'].tolist()
smiles_tst = df.loc[df['split']=='test', 'smiles'].tolist()

# make tensor dataset that transformed with tokenizing and encoding
dataset_trn = StringDataset(smiles_voc, smiles_trn)
dataset_vld = StringDataset(smiles_voc, smiles_vld)
dataset_tst = StringDataset(smiles_voc, smiles_tst)

# fcd metrics
from fcd import get_fcd, load_ref_model, canonical_smiles
fcd_model = load_ref_model()
def calculate_fcd_metrics(smi_list_1, smi_list_2):
	# randome choice of 10000 molecules (at least 10000)
	sample1 = np.random.choice(smi_list_1, 10000, replace=False)
	sample2 = np.random.choice(smi_list_2, 10000, replace=False)

	# get canonical smiles
	can_samp_1 = [x for x in canonical_smiles(sample1) if x is not None]
	can_samp_2 = [x for x in canonical_smiles(sample2) if x is not None]

	# calculation of FCD
	fcd_score = get_fcd(can_samp_1, can_samp_2, fcd_model)

	return fcd_score


# get loss info
import glob
log = glob.glob('24*.log')[0]
print(log)
with open(log, 'r') as f:
	loss_ls, bce_ls, kld_ls = [], [], []
	for line in f:
		if re.search('epoch loss', line):
			x = line.rstrip().split()
			loss = float(x[7].split(':')[-1].replace(',',''))
			bce = float(x[8].split(':')[-1].replace(',',''))
			kld = float(x[9].split(':')[-1].replace(',',''))

			loss_ls.append(loss)
			bce_ls.append(bce)
			kld_ls.append(kld)

# from 20 to 200 by 20
#ep_ls = [20*i for i in range(1,11)]
ep_ls = [20*i for i in range(5,11)]
#ep_ls = [10*i for i in range(5,21)]
info = {'epoch':[], 'reconstruction_ratio':[], 'validity':[], 'uniqueness':[], 'novelty':[], 'int_div':[], 'fcd':[]}
for ep in ep_ls:
	print(f'\nepoch-{ep} is processing...')
	ckpt_file = f'{base_dir}/models/vae_e{ep}.ckpt'
	ckpt = torch.load(ckpt_file)
	VAE_model = LangVAE.construct_by_ckpt_dict(ckpt, VocClass=SmilesVocabulary, EncClass=LSTMEncoder, DecClass=LSTMDecoder, dev=dev)

	# reconstruct smiles
	recon_smiles_trn, _, _ = VAE_model.reconstruct(dataset_trn, dl_njobs=32)
	#recon_smiles_vld, _, _ = VAE_model.reconstruct(dataset_vld, dl_njobs=32)
	#recon_smiles_tst, _, _ = VAE_model.reconstruct(dataset_tst, dl_njobs=32)

	# reconstruction ratio
	rs_trn = (np.array(recon_smiles_trn) == np.array(smiles_trn)).sum() / len(smiles_trn)
	#rs_vld = (np.array(recon_smiles_vld) == np.array(smiles_vld)).sum() / len(smiles_vld)
	#rs_tst = (np.array(recon_smiles_tst) == np.array(smiles_tst)).sum() / len(smiles_tst)


	print(f'Reconstruction ratio of trn: {rs_trn}')
	#print(f'Reconstruction ratio of vld: {rs_vld}')
	#print(f'Reconstruction ratio of tst: {rs_tst}')

	# metrics
	samples, _ = VAE_model.randn_samples(10000)
	trn_set = set(smiles_trn)
	stdmet_trn = standard_metrics(samples, trn_set=trn_set, subs_size=10000, n_jobs=32)

	print("performance - train")
	print(stdmet_trn)

	# loss info
	print('Loss info')
	print(f'loss_sum: {loss_ls[ep-1]}')
	print(f'loss_bce: {bce_ls[ep-1]}')
	print(f'loss_kld: {kld_ls[ep-1]}')

	# fcd metric
	fcd_score = calculate_fcd_metrics(smiles_trn, samples)
	print('FCD score')
	print(f': {fcd_score}')

	# df 
	info['epoch'].append(ep)
	info['reconstruction_ratio'].append(rs_trn)
	info['validity'].append(stdmet_trn['validity'])
	info['uniqueness'].append(stdmet_trn['uniqueness'])
	info['novelty'].append(stdmet_trn['novelty'])
	info['int_div'].append(stdmet_trn['intdiv'])
	info['fcd'].append(fcd_score)

# make data frame
info = pd.DataFrame(info)
info.to_csv('performace.tsv', sep='\t', index=False)
print('\n\nCleaned data')
print(info)
