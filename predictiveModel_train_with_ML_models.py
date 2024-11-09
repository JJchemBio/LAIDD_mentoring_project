'''
Written by Jinhoon Jeong (Dong-A ST), as a participant of the LAIDD mentoring project
'''
import pandas as pd
import numpy as np
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
import pickle, os
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import umap
import matplotlib.pyplot as plt

# SGK1 binding assay data
data = pd.read_csv('./data/BindingDB_SGK1_IC50_duplicatesRemoved.tsv', sep='\t')

save_model = False

# make fingerprint
def ECFP(smiles_list):
	errors =[]
	fps=[]
	for i in range(len(smiles_list)):
		smi = smiles_list[i]
		try:
			m = Chem.MolFromSmiles(smi)
			fp = np.array(AllChem.GetMorganFingerprintAsBitVect(m,2,nBits=1024))
			fps.append(fp)
		except Exception as e:
			print(f"Error at index {i} for SMILES {smi}: {e}")
			errors.append(i)
	return fps

fingerprints = ECFP(data['canonical_smiles'].tolist())

# split train / test dataset
from sklearn.model_selection import train_test_split, StratifiedKFold

X = np.array(fingerprints)
y = np.array(-np.log10(data['IC50']*1E-9))

# divide by 10
y_binned = pd.qcut(y, q=7)
y_binned = [str(x) for x in y_binned]

# show stratified results
y_table = {}
for rg in y_binned:
	if rg in y_table:
		y_table[rg] += 1
	else:
		y_table[rg] = 1
for a in sorted(y_table):
	print(a, y_table[a])

#print('X :', X[:10])
#print('y :', y[:10])

train_feature, test_feature, train_label, test_label = train_test_split(X, y, test_size=0.1, random_state=2024, stratify=y_binned)
# split train data into train and validation data with 20-fold
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=2024)
train_label_binned = pd.qcut(train_label, q=7)
train_label_binned = [str(x) for x in train_label_binned]

# umap
mapper = umap.UMAP()
trn_umap = mapper.fit_transform(train_feature)
tst_umap = mapper.transform(test_feature)

_fig = plt.figure(figsize=(5,5))
ax = _fig.add_subplot(111)
ax.scatter(trn_umap[:,0], trn_umap[:,1], s=3.5, alpha=0.55, label='train')
ax.scatter(tst_umap[:,0], tst_umap[:,1], s=3.5, alpha=0.55, color='red', label='test')
plt.xlabel('UMAP1')
plt.ylabel('UMAP2')
plt.legend()
plt.savefig('umap_train_test.png', dpi=600)


if save_model:
	# training function
	def training_and_eval(model, X_train, y_train, X_val, y_val):
		# training
		model.fit(X_train, y_train)
		# predcit validation data
		y_pred = model.predict(X_val)
		# MSE, pcc
		mse = mean_squared_error(y_val, y_pred)
		pcc, _ = pearsonr(y_val, y_pred)
	
		return model, mse, pcc
	
	# training
	from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
	from sklearn.svm import SVR
	import xgboost as xgb
	from sklearn.model_selection import cross_val_score
	
	rf_val_mse_scores, rf_val_pcc_scores, rf_models = [], [], []
	svm_val_mse_scores, svm_val_pcc_scores, svm_models = [], [], []
	GBM_val_mse_scores, GBM_val_pcc_scores, GBM_models = [], [], []
	xgboost_val_mse_scores, xgboost_val_pcc_scores, xgboost_models = [], [], []
	# 20 fold cross validation
	cnt=0
	for train_index, val_index in skf.split(train_feature, train_label_binned):
		cnt+=1
		print(f'\n---------- {cnt}-fold --------------')
		# split train, validation set
		X_train, X_val = train_feature[train_index], train_feature[val_index]
		y_train, y_val = train_label[train_index], train_label[val_index]
	
		#------------------------- RF model -------------------------------
		rf_model = RandomForestRegressor(random_state=2024) 
		
		# training and evaluating validation set
		rf_model, mse, pcc = training_and_eval(rf_model, X_train, y_train, X_val, y_val)
		
		# save validation performance
		rf_val_mse_scores.append(mse)
		rf_val_pcc_scores.append(pcc)
	
		# save model
		rf_models.append(rf_model)
	
		print(f'Random forest model\nMSE: {mse}\nPCC: {pcc}')
	
		#------------------------- SVM model -------------------------------
		svm_model =  SVR(kernel='rbf', C=1.0, epsilon=0.1)
		
		# training and evaluating validation set
		svm_model, mse, pcc = training_and_eval(svm_model, X_train, y_train, X_val, y_val)
		
		# save validation performance
		svm_val_mse_scores.append(mse)
		svm_val_pcc_scores.append(pcc)
	
		# save model
		svm_models.append(svm_model)
		
		print(f'SVM model\nMSE: {mse}\nPCC: {pcc}')
	
		#------------------------- Gradient boositng model -------------------------------
		GBM_model =  GradientBoostingRegressor(random_state=2024)
		
		# training and evaluating validation set
		GBM_model, mse, pcc = training_and_eval(GBM_model, X_train, y_train, X_val, y_val)
		
		# save validation performance
		GBM_val_mse_scores.append(mse)
		GBM_val_pcc_scores.append(pcc)
	
		# save model
		GBM_models.append(GBM_model)
		
		print(f'GBM model\nMSE: {mse}\nPCC: {pcc}')
	
		#------------------------- XGBoost model -------------------------------
		xgboost_model =  xgb.XGBRegressor(objective='reg:squarederror', random_state=2024)
		
		# training and evaluating validation set
		xgboost_model, mse, pcc = training_and_eval(xgboost_model, X_train, y_train, X_val, y_val)
		
		# save validation performance
		xgboost_val_mse_scores.append(mse)
		xgboost_val_pcc_scores.append(pcc)
	
		# save model
		xgboost_models.append(xgboost_model)
		
		print(f'XGBoost model\nMSE: {mse}\nPCC: {pcc}')
	
	# Print the results
	print(f'\n----- RF validation performance -----')
	print("Mean MSE: ", round(np.mean(rf_val_mse_scores),3))
	print("Standard deviation of MSE: ", round(np.std(rf_val_mse_scores),3))
	print("Mean PCC: ", round(np.mean(rf_val_pcc_scores),3))
	print("Standard deviation of PCC: ", round(np.std(rf_val_pcc_scores),3))
	print(f'\n----- SVM validation performance -----')
	print("Mean MSE: ", round(np.mean(svm_val_mse_scores),3))
	print("Standard deviation of MSE: ", round(np.std(svm_val_mse_scores),3))
	print("Mean PCC: ", round(np.mean(svm_val_pcc_scores),3))
	print("Standard deviation of PCC: ", round(np.std(svm_val_pcc_scores),3))
	print(f'\n----- GBM validation performance -----')
	print("Mean MSE: ", round(np.mean(GBM_val_mse_scores),3))
	print("Standard deviation of MSE: ", round(np.std(GBM_val_mse_scores),3))
	print("Mean PCC: ", round(np.mean(GBM_val_pcc_scores),3))
	print("Standard deviation of PCC: ", round(np.std(GBM_val_pcc_scores),3))
	print(f'\n----- XGBoost validation performance -----')
	print("Mean MSE: ", round(np.mean(xgboost_val_mse_scores),3))
	print("Standard deviation of MSE: ", round(np.std(xgboost_val_mse_scores),3))
	print("Mean PCC: ", round(np.mean(xgboost_val_pcc_scores),3))
	print("Standard deviation of PCC: ", round(np.std(xgboost_val_pcc_scores),3))
	
	
	# save models
	os.system('mkdir -p model/ml_model')
	
	def save_models(models, model_name):
		with open(f'model/ml_model/{model_name}.pkl', 'wb') as f:
			pickle.dump(models, f)
	
	save_models(rf_models, 'rf_models')
	save_models(svm_models, 'svm_models')
	save_models(GBM_models, 'GBM_models')
	save_models(xgboost_models, 'xgboost_models')

	# training all data (train-set + validation-set)
	rf_model_final = RandomForestRegressor(random_state=2024)
	rf_model_final.fit(train_feature, train_label)
	save_models(rf_model_final, 'rf_model_final')

	svm_model_final =  SVR(kernel='rbf', C=1.0, epsilon=0.1)
	svm_model_final.fit(train_feature, train_label)
	save_models(svm_model_final, 'svm_model_final')

	GBM_model_final =  GradientBoostingRegressor(random_state=2024)
	GBM_model_final.fit(train_feature, train_label)
	save_models(GBM_model_final, 'GBM_model_final')
	
	xgboost_model_final =  xgb.XGBRegressor(objective='reg:squarederror', random_state=2024)
	xgboost_model_final.fit(train_feature, train_label)
	save_models(xgboost_model_final, 'xgboost_model_final')

# load models
def load_models(model_name):
	with open(f'model/ml_model/{model_name}.pkl','rb') as f:
		models = pickle.load(f)
	return models

rf_model_final = load_models('rf_model_final')
svm_model_final = load_models('svm_model_final')
GBM_model_final = load_models('GBM_model_final')
xgboost_model_final = load_models('xgboost_model_final')

def check_performance_with_testset(model, model_name):
	# check test-set
	y_pred = model.predict(test_feature)
	mse = mean_squared_error(test_label, y_pred)
	pcc, _ = pearsonr(test_label, y_pred)
	
	print(f'\n----- {model_name} test performance -----')
	print("MSE: ", round(mse,3))
	print("PCC: ", round(pcc,3))

	return y_pred

rf_y_pred = check_performance_with_testset(rf_model_final, 'RF')
svm_y_pred = check_performance_with_testset(svm_model_final, 'SVM')
GBM_y_pred = check_performance_with_testset(GBM_model_final, 'GBM')
xgboost_y_pred = check_performance_with_testset(xgboost_model_final, 'XGBoost')

# scatter plot
import seaborn as sns

df = pd.DataFrame({
	'True_values': test_label,
	'RF_model': rf_y_pred,
	'SVM_model': svm_y_pred,
	'GBM_model': GBM_y_pred,
	'XGBoost_model': xgboost_y_pred,
})

fig, axes = plt.subplots(1, 4, figsize=(16,4), sharex=True, sharey=True)

# scatterplot by each models
for i, model in enumerate(['RF_model', 'SVM_model', 'GBM_model', 'XGBoost_model']):
	sns.scatterplot(ax=axes[i], x='True_values', y=model, data=df)
	axes[i].set_xlim(3.8,10)
	axes[i].set_ylim(3.8,10)
	axes[i].set_title(f'Scatterplot for {model}')
	axes[i].set_xlabel('True Binding Affinity')
	axes[i].set_ylabel('Predicted Binding Affinity')

	axes[i].plot(np.linspace(3.8,10), np.linspace(3.8,10),'r--')

plt.tight_layout()
plt.savefig('test_4_model.png')

# In AWS, this will not be executed.
#plt.show()

