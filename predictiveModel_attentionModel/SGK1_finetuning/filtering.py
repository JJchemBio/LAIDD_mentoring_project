'''
Written by Jinhoon Jeong (Dong-A ST), as a participant of the LAIDD mentoring project
All scripts in bs_denovo, aff_pred were provided as part of the mentoring project and include some minor modifications.
'''

import torch
import pandas as pd
import numpy as np
from bs_denovo import bs_chem
from rdkit import RDLogger
import math
RDLogger.DisableLog('rdApp.*')

# level
P1 = True
P2 = False
P3 = False
P4 = False

zinc_df = pd.read_csv('')

if P1:
	#import gen_molecules
	rand_gen_df = pd.read_csv("postproc/random_gen.csv")
	guid_gen_df = pd.read_csv("postproc/guided_gen.csv")
	
	# random generation validity
	rand_gen_df['valid'] = rand_gen_df['gen'].apply(lambda x: int(bs_chem.is_valid_smiles(x)))
	print("random gen valid count: ", rand_gen_df['valid'].sum())
	# guide generation validity
	guid_gen_df['valid'] = guid_gen_df['gen'].apply(lambda x: int(bs_chem.is_valid_smiles(x)))
	print("guided gen valid count: ", guid_gen_df['valid'].sum())
	
	# add canonical smiles to 'gen_cs' column
	rand_gen_cs = []
	for row in rand_gen_df.itertuples():
		if row.valid == 0:
			rand_gen_cs.append("<>")
		else:
			rand_gen_cs.append(bs_chem.convert_to_canon(row.gen, verbose=True))
	rand_gen_df['gen_cs'] = rand_gen_cs
	
	guid_gen_cs = []
	for row in guid_gen_df.itertuples():
		if row.valid == 0:
			guid_gen_cs.append("<>")
		else:
			guid_gen_cs.append(bs_chem.convert_to_canon(row.gen, verbose=True))
	guid_gen_df['gen_cs'] = guid_gen_cs
	
	# load SGK1 binding affinity data
	ba_df = pd.read_csv("./BindingDB_SGK1_IC50_duplicatesRemoved.tsv", sep='\t')	# binding affinity dataframe
	ba_df['pIC50'] = ba_df['IC50'].apply(lambda x: -math.log10(x*1E-9))
	known_actives = ba_df['canonical_smiles'].tolist()
	
	# check novelty
	# set of known actives
	ka_set = set(known_actives)
	
	# random generation novel
	rand_gen_df['novel'] = rand_gen_df['gen_cs'].apply(lambda x: int((x not in ka_set) and (x != "<>")))
	print("\nrandom gen novel count: ", rand_gen_df['novel'].sum())
	# guide generation novel
	guid_gen_df['novel'] = guid_gen_df['gen_cs'].apply(lambda x: int((x not in ka_set) and (x != "<>")))
	print("guided gen novel count: ", guid_gen_df['novel'].sum())
	
	# save updated DF
	rand_gen_df.to_csv("postproc/random_gen.csv", index=False)
	guid_gen_df.to_csv("postproc/guided_gen.csv", index=False)

if P2:
	# check vality and novelty
	rand_p1_df = rand_gen_df[(rand_gen_df['valid'] == 1) & (rand_gen_df['novel'] == 1)]
	print('\n# of novel and valid mols (random):', rand_p1_df.shape[0])
	guid_p1_df = guid_gen_df[(guid_gen_df['valid'] == 1) & (guid_gen_df['novel'] == 1)]
	print('# of novel and valid mols (guided):', guid_p1_df.shape[0])
	
	# remove duplicates
	rand_p1_df = rand_p1_df.loc[rand_p1_df['gen_cs'].drop_duplicates().index]
	print('\nAfter removing duplicates (random): ',rand_p1_df.shape[0])
	guid_p1_df = guid_p1_df.loc[guid_p1_df['gen_cs'].drop_duplicates().index]
	print('After removing duplicates (guided): ', guid_p1_df.shape[0])
	
	# RDKit Mol objects of generated compounds
	rand_p1_mols = rand_p1_df['gen_cs'].apply(lambda x: bs_chem.get_mol(x)).tolist()
	guid_p1_mols = guid_p1_df['gen_cs'].apply(lambda x: bs_chem.get_mol(x)).tolist()
	
	# remove None
	rand_p1_df = rand_p1_df.loc[pd.notna(rand_p1_mols), :]
	guid_p1_df = guid_p1_df.loc[pd.notna(guid_p1_mols), :]
	rand_p1_mols = [x for x in rand_p1_mols if x != None]
	guid_p1_mols = [x for x in guid_p1_mols if x != None]
	
	# calculates QED
	rand_p1_df['QED'] = bs_chem.get_QEDs(rand_p1_mols)
	guid_p1_df['QED'] = bs_chem.get_QEDs(guid_p1_mols)
	
	# calculates SA
	rand_p1_df['SAS'] = bs_chem.get_SASs(rand_p1_mols)
	guid_p1_df['SAS'] = bs_chem.get_SASs(guid_p1_mols)
	
	# P1 filtering applied, and now saved
	rand_p1_df.to_csv("postproc/random_gen_P1.csv", index=False)
	guid_p1_df.to_csv("postproc/guided_gen_P1.csv", index=False)
	
	# filtering by QED
	rand_p2_df = rand_p1_df[(rand_p1_df['QED'] > 0.5)]
	print('\nAfter filtering based on QED(random): ', rand_p2_df.shape)
	guid_p2_df = guid_p1_df[(guid_p1_df['QED'] > 0.5)]
	print('After filtering based on QED (guided): ', guid_p2_df.shape)
	
	# filtering by SA
	rand_p2_df = rand_p2_df[(rand_p2_df['SAS'] < 4.0)]
	print('\nAfter filtering based on SAS (random): ', rand_p2_df.shape)
	guid_p2_df = guid_p2_df[(guid_p2_df['SAS'] < 4.0)]
	print('After filtering based on SAS (guided): ', guid_p2_df.shape)
	
	# save P2
	rand_p2_df.to_csv("postproc/random_gen_P2.csv", index=False)
	guid_p2_df.to_csv("postproc/guided_gen_P2.csv", index=False)


if P3:
	import pandas as pd
	from aff_pred.dl_model import DL_model, ensemble_inference
	from aff_pred import dl_data
	
	dev = 'cuda'
	sgk1_seq = 'MTVKTEAAKGTLTYSRMRGMVAILIAFMKQRRMGLNDFIQKIANNSYACKHPEVQSILKISQPQEPELMNANPSPPPSPSQQINLGPSSNPHAKPSDFHFLKVIGKGSFGKVLLARHKAEEVFYAVKVLQKKAILKKKEEKHIMSERNVLLKNVKHPFLVGLHFSFQTADKLYFVLDYINGGELFYHLQRERCFLEPRARFYAAEIASALGYLHSLNIVYRDLKPENILLDSQGHIVLTDFGLCKENIEHNSTTSTFCGTPEYLAPEVLHKQPYDRTVDWWCLGAVLYEMLYGLPPFYSRNTAEMYDNILNKPLQLKPNITNSARHLLEGLLQKDRTKRLGAKDDFMEIKSHVFFSLINWDDLINKKITPPFNPNVSGPNDLRHFDPEFTEEPVPNSIGKSPDSVLVTASVKEAAEAFLGFSYAPPTDSFL'
	THRESHOLD = 6.0
	
	rand_p2_df = pd.read_csv("postproc/random_gen_P2.csv")
	guid_p2_df = pd.read_csv("postproc/guided_gen_P2.csv")
	
	evaluation = True
	if evaluation:
		rand_smiles = rand_p2_df['gen_cs'].tolist()
		guid_smiles = guid_p2_df['gen_cs'].tolist()
		
		# load saved binding affinity prediction models (10 models)
		mpath = './epoch_100/model/finetuned_model/SGK1_model_GCN{}'
		dl_models = []
		for mid in range(10):
			print(f'{mid} model is evaluating...')
			dl_model_info = torch.load(mpath.format(mid), map_location=dev, weights_only=False)
			ba_predictor = DL_model()
			ba_predictor.to(dev)
			ba_predictor.load_state_dict(dl_model_info['State_dict'])
			ba_predictor.eval()
			dl_models.append(ba_predictor)
		
		# random generation SGK1 pIC50 prediction
		rand_preds = ensemble_inference(dl_models, rand_smiles, sgk1_seq, bsz=100, dev=dev)
		print(rand_preds.shape)
		
		# guided generation SGK1 pIC50 prediction
		guid_preds = ensemble_inference(dl_models, guid_smiles, sgk1_seq, bsz=100, dev=dev)
		print(guid_preds.shape)
		
		rand_p2_df['pred'] = rand_preds
		guid_p2_df['pred'] = guid_preds
		
		# saving updated DF
		rand_p2_df.to_csv("postproc/random_gen_P2.csv", index=False)
		guid_p2_df.to_csv("postproc/guided_gen_P2.csv", index=False)
	
	rand_p3_df = rand_p2_df[rand_p2_df['pred'] > THRESHOLD]
	print(rand_p3_df.shape)
	guid_p3_df = guid_p2_df[guid_p2_df['pred'] > THRESHOLD]
	print(guid_p3_df.shape)
	
	# save P3
	rand_p3_df.to_csv("postproc/random_gen_P3.csv", index=False)
	guid_p3_df.to_csv("postproc/guided_gen_P3.csv", index=False)

# evaluation specificity
if P4:
	import pandas as pd
	from aff_pred.dl_model import DL_model, build_inference_input
	from aff_pred import dl_data
	from torch.utils import data
	
	save_df = False
	if save_df:
		# take all protein sequence
		assay = pd.read_csv('../kinase_data_wo_SGK1.csv')
		seqs = assay['seq'].tolist()
		seqs = list(set(seqs))
		print('Total num of proteins: ', len(seqs))

		dev = 'cuda'

		rand_p3_df = pd.read_csv("postproc/random_gen_P3.csv")
		guid_p3_df = pd.read_csv("postproc/guided_gen_P3.csv")

		rand_smiles = rand_p3_df['gen_cs'].tolist()
		guid_smiles = guid_p3_df['gen_cs'].tolist()
		all_smiles = rand_smiles + guid_smiles

		# make unified df
		df = pd.DataFrame({'smi':all_smiles})

		# load saved binding affinity prediction model (pretrained)
		mpath = '../model/pretrained_model/kinase_wo_SGK1_model_GCN.pt'
		dl_model_info = torch.load(mpath, map_location=dev, weights_only=False)
		model = DL_model()
		model.to(dev)
		model.load_state_dict(dl_model_info['State_dict'])
		model.eval()

		# get BA
		idx_list = []
		for h, prot_seq in enumerate(seqs):
			print(f'{h+1} / {len(seqs)}')
			inp_data = build_inference_input(all_smiles, prot_seq)
			dldr = data.DataLoader(inp_data, batch_size=99999, shuffle=False, collate_fn=dl_data.collate)
			for bat in dldr:
				drugs, targets, labels, seq_lens = bat  # labels are not used
				drugs = drugs.to(dev)
				atom_feats = drugs.ndata['h']
				targets = targets.to(dev)
				seq_lens = seq_lens.to(dev)
		
				output = model(drugs, atom_feats, targets, seq_lens, dev)  # (n_mols, 1) tensor
				output = output.squeeze().tolist()

				prot_id = f'P{h}'
				df[prot_id] = output

		df.to_csv('P3_get_BAs_w_all_prots.csv', index=False)
	
	df = pd.read_csv('P3_get_BAs_w_all_prots.csv')

	# Filtering if 90% of BA values are below 6
	pIC50_threshold = 6
	percentage_threshold = 80
	
	ba_columns = [col for col in df.columns if col.startswith('P')]
	df['below_threshold_percentage'] = (df[ba_columns] < pIC50_threshold).sum(axis=1) / len(ba_columns) * 100

	# Filter compounds where the percentage of BA values below threshold is greater than or equal to the desired percentage
	filtered_df = df[df['below_threshold_percentage'] >= percentage_threshold]
	print('finally filtered: ', filtered_df.shape)

	filtered_df.to_csv('P4_finally_filtered.csv', index=False)
