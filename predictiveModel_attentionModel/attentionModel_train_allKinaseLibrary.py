'''
Written by Jinhoon Jeong (Dong-A ST), as a participant of the LAIDD mentoring project
Code lines realted to the attention model were mostly referred to the original code provided from the project.
'''
import torch, math, os
from torch import nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from scipy import stats	
from torch.utils.data import DataLoader
import pickle
import json
import dgl
import GCN_model
import time

# setting
path = './'
os.system('mkdir -p data/train_data data/val_data data/test_data model/pretrained_model model/finetuned_model')
os.system('mkdir -p results')
data_processing = False
training		= False
evaluation_test = True

#hyperparameters
BATCH_SIZE= 256
LR= 0.001
NUM_EPOCHS=500
patience = 20
	
if torch.cuda.is_available():
	print('use GPU')
	device = 'cuda'
else:
	print('use CPU')
	device = 'cpu'

def collate(sample):
	'''
	input : sample
		sample -> tuple list
		- graph : DGL graph
		- proteins : protein sequence
		- labels : affinity
		- seq_lens : sequence length

	output : batched_graph, torch.tensor(proteins), torch.tensor(labels), torch.tensor(seq_lens)

	'''
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
	print('data processing...')
	
	# read assay
	# ,canonical_smiles,prot_id,med_pic50,prim_acc,target_chembl_id,assay_variant_mutation,corrected_mutation,seq
	assay = pd.read_csv(f'{path}/kinase_data_wo_SGK1.csv', sep=',')
	
	# get row indexes
	indicies = assay.index
	
	# X, y
	X = np.array(assay['canonical_smiles'])
	y = np.array(assay['med_pic50'])
	
	# split train / test dataset
	train_index, test_index = train_test_split(indicies, test_size=0.2, random_state=2024)
	train_df = assay.loc[train_index].reset_index(drop=True)
	test_df = assay.loc[test_index].reset_index(drop=True)
	
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
	
	train_df.to_csv(f'{path}/data/train_data/kinase_wo_SGK1_train.csv')
	test_df.to_csv(f'{path}/data/test_data/kinase_wo_SGK1_test.csv')
	
	#train data processing
	train_g = [dgl.add_self_loop(smiles_to_bigraph(smi, node_featurizer=atom_featurizer, edge_featurizer=bond_featurizer)) for smi in train_df['canonical_smiles']]
	seq_cutNnumbering = [seq_cat(prot) for prot in train_df['seq']]
	seq_lens = [len(prot) for prot in train_df['seq']]
	train_y = [x for x in train_df['med_pic50']]
	train_data = list(zip(train_g, seq_cutNnumbering, train_y, seq_lens))
	
	#test data processing
	test_g = [dgl.add_self_loop(smiles_to_bigraph(smi, node_featurizer=atom_featurizer, edge_featurizer=bond_featurizer)) for smi in test_df['canonical_smiles']]
	seq_cutNnumbering = [seq_cat(prot) for prot in test_df['seq']]
	seq_lens = [len(prot) for prot in test_df['seq']]
	test_y = [x for x in test_df['med_pic50']]
	test_data = list(zip(test_g, seq_cutNnumbering, test_y, seq_lens))
	
	# save as pickle 
	with open(f'{path}/data/train_data/kinase_wo_SGK1_train.pickle','wb') as fw:
		pickle.dump(train_data,fw)
	with open(f'{path}/data/test_data/kinase_wo_SGK1_test.pickle','wb') as fw:
		pickle.dump(test_data,fw)
	
	print('Done')


if training:
	#학습 데이터 로드
	with open(f'{path}/data/train_data/kinase_wo_SGK1_train.pickle','rb') as fr:
		train_data = pickle.load(fr)
	with open(f'{path}/data/test_data/kinase_wo_SGK1_test.pickle','rb') as fr:
		test_data = pickle.load(fr)
	
	#Dataloader 생성
	train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate, drop_last=False)
	test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate, drop_last=False)
	
	#모델 초기화 및 GPU로 업로드
	model = GCN_model.DL_model()
	model = model.to(device)
	
	#사전학습된 모델 로드
	#pretrained_model_path = f'{path}/model/pretrained_model/model_GCN.pt'
	#model_info = torch.load(pretrained_model_path)
	#model.load_state_dict(model_info['State_dict'])
	
	# make pretrained model
	model_path = f'{path}/model/pretrained_model/kinase_wo_SGK1_model_GCN.pt'
	
	#loss 함수 및 optimizer 설정
	loss_fn = nn.MSELoss()
	#optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0.00001)
	optimizer = torch.optim.Adam(model.parameters(), lr=LR)
	
	#loss 추적을 위한 리스트 초기화
	train_losses = []
	avg_train_losses = []  # epoch당 평균 train loss
	avg_test_losses = [] #epoch당 평균 test loss
	
	best_mse = 1000 #MSE 저장
	counter = 0 #early stopping을 위한 카운터
	epoch_check = 0

	f = open('log', 'w', buffering=1)
	t1 = time.time()
	
	for epoch in range(NUM_EPOCHS):
		print(f'\nEpoch {epoch}...')
		#모델을 학습 모드로 설정
		model.train()
		
		t3 = time. time()
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
		t4 = time. time()
		print(f'Elapsed time of 1 epoch: {round(t4-t3, 3)} sec')
	
		#평균 train loss 계산, test loss 및 pcc 계산
		train_loss = np.average(train_losses)
		mse_test, pcc_test,_, _ =evaluation(model,test_loader)
	
		# 평균 loss 추적
		avg_train_losses.append(train_loss)
		avg_test_losses.append(mse_test)
	
		#Epoch 진행 상황 출력
		print_msg = (f'[{epoch + epoch_check}/{NUM_EPOCHS + epoch_check}] ' +
					 f'train_loss: {train_loss:.5f} ' +
					 f'test_loss: {mse_test:.5f}')
	
		print(print_msg)
		f.write(f'{print_msg}\n')
	
		#train loss 리스트 초기화
		train_losses = []
	
		#best mse 값 갱신
		if best_mse > mse_test:
			counter = 0  #counter 초기화
			print('best mse renew : ' + str(best_mse) + ' --> ' + str(mse_test))
			best_mse = mse_test
	
			#모델 상태 저장
			state = {'Epoch': epoch,
					 'State_dict': model.state_dict(),
					 'optimizer': optimizer.state_dict(),
					 'test MSE': mse_test,
					 'avg_train_losses': avg_train_losses,
					 'avg_test_losses': avg_test_losses}
			torch.save(state, model_path)
	
		else:
			#early stopping 카운터 증가
			counter = counter + 1
			print('Early Stopping counter : ', str(counter))
	
			#patience 초과시 학습 중단
			if counter > patience:
				break
	t2 = time.time()
	print(f'Total elapsed time: {t2-t1} sec')
	f.write(f'Total elapsed time: {t2-t1} sec\n')
	f.close()

if evaluation_test:
	import seaborn as sns
	import matplotlib.pyplot as plt

	#테스트 데이터 로드
	with open(f'{path}/data/test_data/kinase_wo_SGK1_test.pickle','rb') as fr:
		test_data = pickle.load(fr)
	test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate, drop_last=False)

	# load model
	model_path = f'{path}/model/pretrained_model/kinase_wo_SGK1_model_GCN.pt'
	model_info = torch.load(model_path)
	model = GCN_model.DL_model()
	model.to(device)
	model.load_state_dict(model_info['State_dict'])

	# evaluation
	mse_test, pcc_test, y_label, y_pred = evaluation(model,test_loader)

	print("MSE: ", round(mse_test ,3))
	print("PCC: ", round(pcc_test ,3))

	with open(f'test_performance.txt', 'w') as f:
		f.write(f'MSE: {mse_test}\n')
		f.write(f'PCC: {pcc_test}\n')

	df = pd.DataFrame({
		'True Values': y_label,
		'Pre-trained model': y_pred
	})

	fig, axe = plt.subplots(1,1, figsize=(4,4))

	sns.scatterplot(ax=axe, x='True Values', y='Pre-trained model', data=df, s=1)
	
	axe.set_xlim(3.1,11)
	axe.set_ylim(3.1,11)
	axe.set_title(f'Test set (n=34,209)', fontsize=20)
	axe.set_xlabel('True Binding Affinity', fontsize=15)
	axe.set_ylabel('Predicted Binding Affinity', fontsize=15)

	axe.plot(np.linspace(3.1,11), np.linspace(3.1,11),'r--', lw=0.7)

	axe.text(4, 9.5, 'MSE = 0.53\nPCC = 0.82', fontsize=12, color='black')

	plt.tight_layout()
	plt.savefig(f'GCN_pretrained_test_eval.png')
	plt.show()