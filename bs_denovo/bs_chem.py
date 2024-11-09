"""
	This is a renewed version of the old chemistry.py.
	The default canonicalization of SMILES allows isomericSmiles=True.
"""

from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, AllChem, MACCSkeys, rdFMCS, Draw
from rdkit.Chem.Scaffolds import MurckoScaffold
import numpy as np
import functools
from typing import List, Callable
from bs_denovo import sascorer
from bs_denovo.py_core import multiproc_task_on_list, pairwise_tupled_ops

def ic50nm_to_pic50(x):
	return -np.log10(x * (10**-9))

def is_valid_smiles(smi):
	mol = Chem.MolFromSmiles(smi)
	if mol == None: return False
	return True

def convert_to_canon(smi, iso=True, verbose=None):
	mol = Chem.MolFromSmiles(smi)
	if mol == None:
		if verbose: print('[ERROR] cannot parse: ', smi)
		return None
	return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=iso)

def get_mol(smi):
	# make MolFromSmiles() picklable
	mol = Chem.MolFromSmiles(smi)
	return mol

def get_mols(smilist, n_jobs=1):
	return multiproc_task_on_list(get_mol, smilist, n_jobs)

def get_valid_canons(smilist, iso=True, n_jobs=1):
	''' multiprocess of smiles canon allowing isomerism '''
	get_canon = functools.partial(convert_to_canon, iso=iso)
	canons = multiproc_task_on_list(get_canon, smilist, n_jobs)
	canons = np.array(canons)
	invalid_ids = np.where(canons==None)[0]
	# insert error string to invalid positions
	canons[invalid_ids] = "<ERR>"

	# Re-checking the parsed smiles, since there are bugs in rdkit parser.
	# https://github.com/rdkit/rdkit/issues/4701
	is_valid = multiproc_task_on_list(is_valid_smiles, canons, n_jobs)
	is_valid = np.array(is_valid)
	invalid_ids = np.where(is_valid==False)[0]
	return np.delete(canons, invalid_ids), invalid_ids

def get_morganfp_by_smi(smi, r=2, b=2048):
	mol = Chem.MolFromSmiles(smi)
	fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=r, nBits=b)
	return fp

def get_mgfps_from_smilist(smilist, r=2, b=2048, n_jobs=1):
	get_fp = functools.partial(get_morganfp_by_smi, r=r, b=b)
	fps = multiproc_task_on_list(get_fp, smilist, n_jobs)
	return fps

def get_maccs_by_smi(smi):
	mol = Chem.MolFromSmiles(smi)
	return MACCSkeys.GenMACCSKeys(mol)

def get_maccs_from_smilist(smilist, n_jobs=1):
	fps = multiproc_task_on_list(get_maccs_by_smi, smilist, n_jobs)
	return fps

def rdk2npfps(rdkfp_list, n_jobs=1):
	""" 
		rdkfp_list: list of RDKit Fingerprint objects 
		Using multi processing manually showed faster operations.
			e.g. for 100k mols, simple np.array(fps_list) gives 126 secs.
			using this function with n_jobs=20 gives 13 secs.
	"""
	arr_list = multiproc_task_on_list(np.array, rdkfp_list, n_jobs)
	return np.array(arr_list)

def np2rdk(npfp):
	bitstring = "".join(npfp.astype(str))
	return DataStructs.cDataStructs.CreateFromBitString(bitstring)

def np2rdkfps(npfps, n_jobs=1):
	rdkfps = multiproc_task_on_list(np2rdk, npfps, n_jobs)
	return rdkfps

# molecular weights MW
def get_mw(mol):
	return Descriptors.ExactMolWt(mol)
def get_MWs(mols, n_jobs=1):
	return multiproc_task_on_list(get_mw, mols, n_jobs)

# QED
def get_QEDs(mols, n_jobs=1):
	return multiproc_task_on_list(Chem.QED.qed, mols, n_jobs)

# SAS
def get_SASs(mols, n_jobs=1):
	return multiproc_task_on_list(sascorer.calculateScore, mols, n_jobs)

# logP
def get_logp(mol):
	return Descriptors.MolLogP(mol)
def get_logPs(mols, n_jobs=1):
	return multiproc_task_on_list(get_logp, mols, n_jobs)

# TPSA
def get_tpsa(mol):
	return Descriptors.TPSA(mol)
def get_TPSAs(mols, n_jobs=1):
	return multiproc_task_on_list(get_tpsa, mols, n_jobs)

# 6 elements in AtomRing 6 -> Hex, 5 -> Pent, 4 -> Quad, 3 -> Tri
def get_ring_counts(mol):
	ring_counts = {'Hex':0, 'Pent':0, 'Quad':0, 'Tri':0, 'others':0}
	rin = mol.GetRingInfo()
	for ring in rin.AtomRings():
		if len(ring) == 6: ring_counts['Hex'] += 1
		elif len(ring) == 5: ring_counts['Pent'] += 1
		elif len(ring) == 4: ring_counts['Quad'] += 1
		elif len(ring) == 3: ring_counts['Tri'] += 1
		else: ring_counts['others'] += 1
	return ring_counts

def tansim_tup(tup):
	# tup is a tuple of (fp1, fp2)
	return DataStructs.FingerprintSimilarity(tup[0], tup[1])

def tvsim_tup(tup, a=0.5, b=0.5):
	# Tversky similarity of tuple of fingerprints
	return DataStructs.TverskySimilarity(tup[0], tup[1], a, b)

def get_pw_simmat(fps1, fps2, sim_tup:Callable, n_jobs=1):
	""" 
		fps1, fps2: list of fingerprint objects 
		sim_tup: similarity operation with tuple input. For tvsim_tup(), you should use functools.partial beforehand.
		> return: pairwise similarity matrix, where rows are fps1, and cols are fps2
	"""
	py_simmat = pairwise_tupled_ops(sim_tup, fps1, fps2, n_jobs)
	return np.array(py_simmat)

# Murcko Scaffold mol list
def get_MrkScfs(mols, n_jobs=1):
	return multiproc_task_on_list(MurckoScaffold.GetScaffoldForMol, mols, n_jobs)

def get_mcs_smarts(pair:tuple, completeRingsOnly=True):
	"""
	https://bertiewooster.github.io/2022/10/09/RDKit-find-and-highlight-the-maximum-common-substructure-between-molecules.html

		pair is a tuple of two mols from rdkit.
		There could be other parameters for configuring MCS, but it is hard to find any instructions.
		- returns:
			mcs.smartsString is SMARTS of mcs
			match1 is substruct matched for pair[0]
			match2 is substruct matched for pair[1]
	"""
	mcs = rdFMCS.FindMCS(pair, completeRingsOnly=completeRingsOnly, timeout=1)
	mcs_mol = Chem.MolFromSmarts(mcs.smartsString)
	match1 = pair[0].GetSubstructMatch(mcs_mol)
	match2 = pair[1].GetSubstructMatch(mcs_mol)
	return (mcs.smartsString, match1, match2)

def get_mcs_pairwise(mol_list1, mol_list2, completeRingsOnly=True, n_jobs=1):
	mcs_op = functools.partial(get_mcs_smarts, completeRingsOnly=completeRingsOnly)
	return pairwise_tupled_ops(mcs_op, mol_list1, mol_list2, n_jobs)

def draw_mols_grid(mols:List, **kwargs):
	"""
		> Examples:
			img1 = Draw.MolsToGridImage(draw_mols, highlightAtomLists=match_info_df['mol_mat'].tolist(), 
							legends=draw_legs, molsPerRow=4, subImgSize=(250,150), useSVG=True)
			Note that elements in legends should be all strings.

		> To save the returned image:
			with open('./scaff_match_list.svg', 'w') as f:
				f.write(img1.data)
	"""
	return Draw.MolsToGridImage(mols, **kwargs)

def mol_scaf_match(mol_smi, scaf_smi, iso=True):
	mol_csm = convert_to_canon(mol_smi, iso=iso)
	scaf_csm = convert_to_canon(scaf_smi, iso=iso)
	if mol_csm is None or scaf_csm is None: return None
	mol_m = Chem.MolFromSmiles(mol_csm)
	scaf_m = Chem.MolFromSmiles(scaf_csm)
	mcsrt, mol_mat, scaf_mat = get_mcs_smarts((mol_m, scaf_m))

	mcs_mol = Chem.MolFromSmarts(mcsrt)
	mol_sm = Chem.MolFromSmarts(mol_csm)
	mol_sm.UpdatePropertyCache()  ## https://github.com/rdkit/rdkit/issues/1596
	scaf_sm = Chem.MolFromSmarts(scaf_csm)
	mat_dict = {
		'hsm':int(mol_sm.HasSubstructMatch(scaf_sm)),
		'mcs_smarts':mcsrt, 'mol_mat':mol_mat, 'scaf_mat':scaf_mat,
		'scaf_fill':np.round(mcs_mol.GetNumAtoms()/scaf_m.GetNumAtoms(), 4).item()
	}
	return mat_dict
