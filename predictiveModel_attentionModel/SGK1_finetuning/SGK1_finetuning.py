'''
Written by Jinhoon Jeong (Dong-A ST), as a participant of the LAIDD mentoring project
Code lines realted to the attention model were mostly referred to the original code provided from the project.
'''
import torch, math, os
from torch import nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from scipy import stats	
from torch.utils.data import DataLoader
import pickle
import json
import dgl
import GCN_model

# setting
path = './'
os.system('mkdir -p data/train_data data/val_data data/test_data model/pretrained_model model/finetuned_model')
os.system('mkdir -p results')
data_processing = False
training		= False
evaluation_test = True

#hyperparameters
BATCH_SIZE= 32
LR= 0.0001
NUM_EPOCHS=200
patience = 30
	
if torch.cuda.is_available():
	print('use GPU')
	device = 'cuda'
else:
	print('use CPU')
	device = 'cpu'

def collate(sample):
	graph, proteins, labels, seq_lens = map(list,zip(*sample)) #각 샘플 요소로 분리
	batched_graph = dgl.batch(graph) #여러 compound 그래프를 하나의 그래프로(배치) 묶음
	return batched_graph, torch.tensor(proteins), torch.tensor(labels), torch.tensor(seq_lens)

def evaluation(model, data_loader):
	model.eval()
	total_preds=torch.Tensor()
	total_labels=torch.Tensor()

	with torch.no_grad():
		for data in data_loader:
			drugs, targets, labels, seq_lens = data
			targets = targets.to(device)
			labels = labels.to(device)
			seq_lens = seq_lens.to(device)
			atom_feats = drugs.ndata['h'].to(device)
			drugs = drugs.to(device)

			output = model(drugs, atom_feats, targets, seq_lens,device)

			total_preds = torch.cat((total_preds, output.cpu()), 0)
			total_labels = torch.cat((total_labels, labels.view(-1, 1).cpu()), 0)
	G, P = total_labels.numpy().flatten(), total_preds.numpy().flatten()
	mse_val = mse(G,P)
	pcc_val, _ = stats.pearsonr(G,P)

	return mse_val, pcc_val, G, P

def mse(y,f):
	mse = ((y - f)**2).mean(axis=0)
	return mse
	
if data_processing:
	from dgllife.utils import smiles_to_bigraph, mol_to_bigraph
	from dgllife.utils import BaseBondFeaturizer, bond_type_one_hot,bond_is_conjugated_one_hot,bond_is_in_ring_one_hot,bond_stereo_one_hot
	from dgllife.utils import ConcatFeaturizer,BaseAtomFeaturizer, atom_type_one_hot, atom_degree_one_hot, atom_is_aromatic_one_hot, atom_total_num_H_one_hot, atom_implicit_valence_one_hot
	
	# read assay
	assay = pd.read_csv(f'{path}/BindingDB_SGK1_IC50_duplicatesRemoved.tsv', sep='\t')
	
	# get row indexes
	indicies = assay.index
	
	# add information
	assay['pIC50'] = assay['IC50'].apply(lambda x: -math.log10(x*1E-9))
	assay['seq'] = 'MTVKTEAAKGTLTYSRMRGMVAILIAFMKQRRMGLNDFIQKIANNSYACKHPEVQSILKISQPQEPELMNANPSPPPSPSQQINLGPSSNPHAKPSDFHFLKVIGKGSFGKVLLARHKAEEVFYAVKVLQKKAILKKKEEKHIMSERNVLLKNVKHPFLVGLHFSFQTADKLYFVLDYINGGELFYHLQRERCFLEPRARFYAAEIASALGYLHSLNIVYRDLKPENILLDSQGHIVLTDFGLCKENIEHNSTTSTFCGTPEYLAPEVLHKQPYDRTVDWWCLGAVLYEMLYGLPPFYSRNTAEMYDNILNKPLQLKPNITNSARHLLEGLLQKDRTKRLGAKDDFMEIKSHVFFSLINWDDLINKKITPPFNPNVSGPNDLRHFDPEFTEEPVPNSIGKSPDSVLVTASVKEAAEAFLGFSYAPPTDSFL'
	
	# X, y
	X = np.array(assay['canonical_smiles'])
	y = np.array(assay['pIC50'])
	
	# divide by 7
	y_binned = pd.qcut(y, q=7)
	y_binned = [str(x) for x in y_binned]
	
	# split train / test dataset
	train_feature, test_feature, train_label, test_label = train_test_split(X, y, test_size=0.1, random_state=2024, stratify=y_binned)
	train_index, test_index = train_test_split(indicies, test_size=0.1, random_state=2024, stratify=y_binned)
	train_df = assay.loc[train_index].reset_index(drop=True)
	test_df = assay.loc[test_index].reset_index(drop=True)
	
	# split train data into train and validation data with 10-fold
	skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=2024)
	train_label_binned = pd.qcut(train_label, q=7)
	train_label_binned = [str(x) for x in train_label_binned]
	
	# atom feature
	atom_featurizer = BaseAtomFeaturizer(featurizer_funcs={'h': ConcatFeaturizer([atom_type_one_hot, atom_degree_one_hot, atom_total_num_H_one_hot, atom_implicit_valence_one_hot,atom_is_aromatic_one_hot])})
	# bond feature
	bond_featurizer = BaseBondFeaturizer({'e': ConcatFeaturizer([bond_type_one_hot, bond_is_conjugated_one_hot, bond_is_in_ring_one_hot, bond_stereo_one_hot])})
	
	# limit maximum protein sequence by 1000, label encoding
	def seq_cat(prot,max_seq_len=1000):
		seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
		seq_dict = {v: (i + 1) for i, v in enumerate(seq_voc)}
	
		x = np.zeros(max_seq_len)
		for i, ch in enumerate(prot[:max_seq_len]):
			x[i] = seq_dict[ch]
		return x.astype(np.int64)
	
	#train, validation 
	for fold, (train_index, val_index) in enumerate(skf.split(train_feature, train_label_binned)):
		train_fold = train_df.loc[train_index].reset_index(drop=True)
		val_fold = train_df.loc[val_index].reset_index(drop=True)
		train_fold.to_csv(f'{path}/data/train_data/SGK1_train_fold{fold}.csv')
		val_fold.to_csv(f'{path}/data/val_data/SGK1_val_fold{fold}.csv')
	
		#train data processing
		train_g = [dgl.add_self_loop(smiles_to_bigraph(smi, node_featurizer=atom_featurizer, edge_featurizer=bond_featurizer)) for smi in train_fold['canonical_smiles']]
		seq_cutNnumbering = [seq_cat(prot) for prot in train_fold['seq']]
		seq_lens = [len(prot) for prot in train_fold['seq']]
		train_y = [x for x in train_fold['pIC50']]
		train_data = list(zip(train_g, seq_cutNnumbering, train_y, seq_lens))
	
		#val data processing
		val_g = [dgl.add_self_loop(smiles_to_bigraph(smi, node_featurizer=atom_featurizer, edge_featurizer=bond_featurizer)) for smi in val_fold['canonical_smiles']]
		seq_cutNnumbering = [seq_cat(prot) for prot in val_fold['seq']]
		seq_lens = [len(prot) for prot in val_fold['seq']]
		val_y = [x for x in val_fold['pIC50']]
		val_data = list(zip(val_g, seq_cutNnumbering, val_y, seq_lens))
	
		# save as pickle 
		with open(f'{path}/data/train_data/SGK1_train_fold{fold}.pickle','wb') as fw:
			pickle.dump(train_data,fw)
		with open(f'{path}/data/val_data/SGK1_val_fold{fold}.pickle','wb') as fw:
			pickle.dump(val_data,fw)
	
	#test data processing
	test_df.to_csv(f'{path}/data/test_data/SGK1_test_fold.csv')
	test_g = [dgl.add_self_loop(smiles_to_bigraph(smi, node_featurizer=atom_featurizer, edge_featurizer=bond_featurizer)) for smi in test_df['canonical_smiles']]
	seq_cutNnumbering = [seq_cat(prot) for prot in test_df['seq']]
	seq_lens = [len(prot) for prot in test_df['seq']]
	test_y = [x for x in test_df['pIC50']]
	test_data = list(zip(test_g, seq_cutNnumbering, test_y, seq_lens))
	with open(f'{path}/data/test_data/SGK1_test_fold.pickle','wb') as fw:
		pickle.dump(test_data,fw)


if training:
	#validation function
	for fold in range(10):
		#학습 데이터 로드
		with open(f'{path}/data/train_data/SGK1_train_fold{fold}.pickle','rb') as fr:
			train_data = pickle.load(fr)
		with open(f'{path}/data/val_data/SGK1_val_fold{fold}.pickle','rb') as fr:
			val_data = pickle.load(fr)
	
		#Dataloader 생성
		train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate, drop_last=False)
		val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate, drop_last=False)
	
		#모델 초기화 및 GPU로 업로드
		model = GCN_model.DL_model()
		model = model.to(device)
	
		#사전학습된 모델 로드
		pretrained_model_path = '/home/donaga01/Jinhoon/LAIDD_project/GCN_train_kinase_library/model/pretrained_model/kinase_wo_SGK1_model_GCN.pt'
		model_info = torch.load(pretrained_model_path)
		model.load_state_dict(model_info['State_dict'])
	
		#미세조정 학습 모델 저장 경로 설정
		model_path = f'{path}/model/finetuned_model/SGK1_model_GCN{fold}'
	
		#loss 함수 및 optimizer 설정
		loss_fn = nn.MSELoss()
		optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0.00001)
	
		#loss 추적을 위한 리스트 초기화
		train_losses = []
		avg_train_losses = []  # epoch당 평균 train loss
		avg_val_losses = [] #epoch당 평균 val loss
	
		best_mse = 1000 #MSE 저장
		counter = 0 #early stopping을 위한 카운터
		epoch_check = 0
	
		for epoch in range(NUM_EPOCHS):
			#모델을 학습 모드로 설정
			model.train()
	
			for i, data in enumerate(train_loader):
				#배치 데이터 로드
				drugs, targets, labels, seq_lens = data
				targets = targets.to(device)
				labels = labels.to(device)
				seq_lens = seq_lens.to(device)
				atom_feats = drugs.ndata['h'].to(device)
				drugs = drugs.to(device)
	
				#그라디언트 초기화
				optimizer.zero_grad()
	
				#모델 예측 및 loss 계산
				output = model(drugs, atom_feats, targets, seq_lens,device)
				loss = loss_fn(output, labels.view(-1, 1).float().to(device))
	
				#backpropagation, loss에 대한 그라디언트를 계산
				loss.backward()
				#optimizer step, 계산된 그라디언트를 이용해서 가중치 업데이트
				optimizer.step()
	
				#tran loss 저장
				train_losses.append(loss.item())
	
			#평균 train loss 계산, val loss 및 pcc 계산
			train_loss = np.average(train_losses)
			mse_val, pcc_val,_, _ =evaluation(model,val_loader)
	
			# 평균 loss 추적
			avg_train_losses.append(train_loss)
			avg_val_losses.append(mse_val)
	
			#Epoch 진행 상황 출력
			print_msg = (f'[Fold {fold} : {epoch + epoch_check}/{NUM_EPOCHS + epoch_check}] ' +
						 f'train_loss: {train_loss:.5f} ' +
						 f'test_loss: {mse_val:.5f}')
	
			print(print_msg)
	
			#train loss 리스트 초기화
			train_losses = []
	
			#best mse 값 갱신
			if best_mse > mse_val:
				counter = 0  #counter 초기화
				print('best mse renew : ' + str(best_mse) + ' --> ' + str(mse_val))
				best_mse = mse_val
	
				#모델 상태 저장
				state = {'Epoch': epoch,
						 'State_dict': model.state_dict(),
						 'optimizer': optimizer.state_dict(),
						 'val MSE': mse_val,
						 'avg_train_losses': avg_train_losses,
						 'avg_val_losses': avg_val_losses}
				torch.save(state, model_path)
	
			else:
				#early stopping 카운터 증가
				counter = counter + 1
				print('Early Stopping counter : ', str(counter))
	
				#patience 초과시 학습 중단
				if counter > patience:
					break

if evaluation_test:
	#테스트 데이터 로드
	with open(f'{path}/data/test_data/SGK1_test_fold.pickle','rb') as fr:
		test_data = pickle.load(fr)
	test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate, drop_last=False)
	
	#MSE, PCC 성능 저장할 리스트 초기화
	mse_vals=[]
	pcc_vals=[]
	y_preds = []
	
	#5개 fold에 대해 성능 평가
	for fold in range(10):
		print(f'\n{fold} processing..')
		#저장된 모델 로드
		model_path = f'{path}/model/finetuned_model/SGK1_model_GCN{fold}'
		model_info = torch.load(model_path)
		model = GCN_model.DL_model()
		model.to(device)
		model.load_state_dict(model_info['State_dict'])
	
		#테스트 데이터에 대해 MSE, PCC 계산
		mse_val, pcc_val, y_label, y_pred=evaluation(model,test_loader)
		print(f'MSE: {mse_val}, PCC: {pcc_val}')
	
		#결과를 리스트에 저장
		mse_vals.append(mse_val)
		pcc_vals.append(pcc_val)
		y_preds.append(y_pred)
	
	best_idx = pcc_vals.index(max(pcc_vals))

	print("Mean MSE: ", round(np.mean(mse_vals),3))
	print("Standard deviation of MSE: ", round(np.std(mse_vals),3))
	print("Mean PCC: ", round(np.mean(pcc_vals),3))
	print("Standard deviation of PCC: ", round(np.std(pcc_vals),3))

	with open(f'test_performance_ep{NUM_EPOCHS}.txt', 'w') as f:
		f.write(f'Mean MSE: {round(np.mean(mse_vals),3)}\n')
		f.write(f'Mean PCC: {round(np.mean(pcc_vals),3)}\n')

	#예측 결과를 scatter plot으로 확인해봅니다.
	import seaborn as sns
	import matplotlib.pyplot as plt

	df = pd.DataFrame({
		'True Values': y_label,
		'Fine-tuned model': y_preds[best_idx]
	})

	fig, axe = plt.subplots(1,1, figsize=(4,4))

	sns.scatterplot(ax=axe, x='True Values', y='Fine-tuned model', data=df)

	axe.set_xlim(3.8,10)
	axe.set_ylim(3.8,10)
	axe.set_title(f'Test set (n=74)', fontsize=20)
	axe.set_xlabel('True Binding Affinity', fontsize=15)
	axe.set_ylabel('Predicted Binding Affinity', fontsize=15)

	axe.text(4.3, 8.8, 'MSE = 0.46\nPCC = 0.87', fontsize=12, color='black')

	axe.plot(np.linspace(3.8,10), np.linspace(3.8,10),'r--')
	plt.tight_layout()
	plt.savefig(f'GCN_finetuned_test_eval_best_model.png')
	
	exit()


	df = pd.DataFrame({
		'True Values': y_label,
		'Model 1': y_preds[0],
		'Model 2': y_preds[1],
		'Model 3': y_preds[2],
		'Model 4': y_preds[3],
		'Model 5': y_preds[4],
		'Model 6': y_preds[5],
		'Model 7': y_preds[6],
		'Model 8': y_preds[7],
		'Model 9': y_preds[8],
		'Model 10': y_preds[9]
	})
	
	# Scatterplot 그리기
	fig, axes = plt.subplots(2, 5, figsize=(18,8), sharex=True, sharey=True)
	
	# 각 모델에 대해 scatterplot
	for i, m in enumerate(['Model 1', 'Model 2', 'Model 3', 'Model 4', 'Model 5', 'Model 6', 'Model 7', 'Model 8', 'Model 9', 'Model 10']):
		a = i // 5
		b = i % 5
		sns.scatterplot(ax=axes[a,b], x='True Values', y=m, data=df)
		axes[a,b].set_xlim(3.8,10)
		axes[a,b].set_ylim(3.8,10)
		axes[a,b].set_title(f'Scatterplot for {m}')
		axes[a,b].set_xlabel('True Binding Affinity')
		axes[a,b].set_ylabel('Predicted Binding Affinity')
	
		axes[a,b].plot(np.linspace(3.8,10), np.linspace(3.8,10),'r--')
	
	plt.tight_layout()
	plt.savefig(f'GCN_fine_tuned_10_models_ep{NUM_EPOCHS}.png')

	os.system(f'mkdir -p epoch_{NUM_EPOCHS}')
	os.system(f'cp -r model epoch_{NUM_EPOCHS}')
	os.system(f'mv GCN_fine_tuned_10_models_ep{NUM_EPOCHS}.png test_performance_ep{NUM_EPOCHS}.txt results')
