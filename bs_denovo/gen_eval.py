from bs_denovo import bs_chem
import numpy as np

def standard_metrics(gen_txt_list, trn_set:set, subs_size, n_jobs=1):
    """
        gen_txt_list: generated text list
        trn_set: set(training smiles used)
        subs_size(k): size of the subset to be used for similarity matrix formation
             - first k samples from gen_txt_list will be used.
    """
    std_mets = {}
    gsize = len(gen_txt_list)
    can_smis, invids = bs_chem.get_valid_canons(gen_txt_list, n_jobs)

    std_mets['validity'] = len(can_smis) / gsize
    uni_smis = list(set(can_smis))
    if len(uni_smis) <= 0:
        std_mets['uniqueness'] = -1
        std_mets['novelty'] = -1
        std_mets['intdiv'] = -1
    else:
        std_mets['uniqueness'] = len(uni_smis) / len(can_smis)
    
        gen_set = set(uni_smis)
        nov_set = gen_set.difference(trn_set)
        std_mets['novelty'] = len(nov_set) / len(uni_smis)

        subs = can_smis[:subs_size]
        sub_fps = bs_chem.get_mgfps_from_smilist(subs)  # use default options
        simmat = bs_chem.get_pw_simmat(sub_fps, sub_fps, sim_tup=bs_chem.tansim_tup, 
                                    n_jobs=n_jobs)  # using tanimoto similarity
        std_mets['intdiv'] = (1-simmat).mean()
    return std_mets