import numpy as np
import torch
import dgl
from dgllife.utils import ConcatFeaturizer,BaseAtomFeaturizer, atom_type_one_hot, atom_degree_one_hot, atom_is_aromatic_one_hot, atom_total_num_H_one_hot, atom_implicit_valence_one_hot
from dgllife.utils import BaseBondFeaturizer, bond_type_one_hot,bond_is_conjugated_one_hot,bond_is_in_ring_one_hot,bond_stereo_one_hot


# atom feature
atom_featurizer = BaseAtomFeaturizer(
    featurizer_funcs={'h': ConcatFeaturizer([atom_type_one_hot, atom_degree_one_hot, atom_total_num_H_one_hot, atom_implicit_valence_one_hot,atom_is_aromatic_one_hot])}
    )
# bond feature
bond_featurizer = BaseBondFeaturizer(
    {'e': ConcatFeaturizer([bond_type_one_hot, bond_is_conjugated_one_hot, bond_is_in_ring_one_hot, bond_stereo_one_hot])}
    )

#단백질 최대 길이 1000으로 자르기, label encoding
def seq_cat(prot,max_seq_len=1000):
    seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
    seq_dict = {v: (i + 1) for i, v in enumerate(seq_voc)}

    x = np.zeros(max_seq_len)
    for i, ch in enumerate(prot[:max_seq_len]):
        x[i] = seq_dict[ch]
    return x.astype(np.int64)

def collate(sample):
    '''
    collate 함수는 딥러닝 모델에 사용할 데이터를 준비하는 역할을 합니다. 개별 데이터 샘플 목록을 받아 모델이 처리하기에 적합한 배치로 묶습니다.

    입력 : sample
        sample은 튜플 목록이며, 각 튜플에는 다음이 포함됩니다.
        - graph : 분자를 나타내는 DGL 그래프
        - proteins : 단백질 서열을 나타내는 텐서
        - labels : 결합친화도 텐서
        - seq_lens : 단백질 서열길이 텐서

    dgl.batch(graph) 를 사용하여 개별 그래프를 단일 배치 그래프로 결합합니다. 이를통해 여러 그래프를 동시에 효율적으로 처리할 수 있습니다.

    출력 : batched_graph, torch.tensor(proteins), torch.tensor(labels), torch.tensor(seq_lens)

    '''
    graph, proteins, labels, seq_lens = map(list,zip(*sample)) #각 샘플 요소로 분리
    batched_graph = dgl.batch(graph) #여러 compound 그래프를 하나의 그래프로(배치) 묶음
    return batched_graph, torch.tensor(proteins), torch.tensor(labels), torch.tensor(seq_lens)