from dgl.nn.pytorch import GraphConv
from dgllife.utils import smiles_to_bigraph
import dgl.backend as F
import torch.nn as nn
import torch
import dgl
from aff_pred import dl_data
import numpy as np
from typing import List
from torch.utils import data

class DL_model(nn.Module):
    def __init__(self, embed_dim=128,  output_dim=128, num_features_xd=68, num_features_xt=25, k_size_xt=8, n_filters=32, dropout=0.1, infinity = -5.0E10):
        super(DL_model, self).__init__()

        #GCN layer, molecule graph feature extraction
        self.conv1 = GraphConv(num_features_xd, embed_dim)
        self.conv2 = GraphConv(embed_dim, embed_dim*2)
        self.conv3 = GraphConv(embed_dim*2, output_dim)

        #Embedding, 1DCNN layer, protein feature extraction
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim, padding_idx = 0)
        self.conv_xt_1 = nn.Conv1d(in_channels=embed_dim, out_channels=n_filters, kernel_size=k_size_xt)
        self.conv_xt_2 = nn.Conv1d(in_channels = n_filters ,out_channels= n_filters*2,kernel_size = k_size_xt)
        self.conv_xt_3 = nn.Conv1d(in_channels =n_filters*2 ,out_channels= output_dim,kernel_size = k_size_xt)

        #Fully connected layer, DTA prediction
        self.fc1 = nn.Linear(2*output_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, 1)

        #Other components
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.inf = infinity

    def graph_in_batch(self, batch_num_objs, h):
        #Pad graph node features to batch size
        hs = F.pad_packed_tensor(h,batch_num_objs,0)

        return hs

    def padding_zero_to_inf(self, similarity_score, batch_num_objs, seq_lens,device):
        #apply masking to similarity scores to ignore padded areas
        num = self.inf

        # mask for graphs and sequences that go beyond actual lengths
        batch_mask = torch.arange(similarity_score.size(1)).expand(len(batch_num_objs), -1).to(device) >= batch_num_objs.unsqueeze(1)
        seq_mask = torch.arange(similarity_score.size(2)).expand(len(seq_lens), -1).to(device) >= seq_lens.unsqueeze(1)

        # Apply masking to similarity_score
        similarity_score.masked_fill_(batch_mask.unsqueeze(2), num)
        similarity_score.masked_fill_(seq_mask.unsqueeze(1), num)

        return similarity_score

    def attention_cal(self, xd, xt, batch_num_objs, seq_lens,device):
        #calculate attention between drug and protein feature

        #compute similarity score between drug and protein
        similarity_score = torch.bmm(xd,xt)

        #apply masking to similarity score
        similarity_score = self.padding_zero_to_inf(similarity_score, batch_num_objs,seq_lens,device )

        #softmax over different dimensions
        m1 = nn.Softmax(dim=1)
        m2 = nn.Softmax(dim=-1)

        s_a = m1(similarity_score)
        a_s = m2(similarity_score)

        #calculate attention for drug
        s_a = s_a.permute(0,2,1)
        s_a_xd = torch.bmm(s_a,xd)
        feature1 = torch.sum(s_a_xd,1)

        #calculate attention for protein
        xt = xt.permute(0,2,1)
        a_s_xt = torch.bmm(a_s,xt)
        feature2 = torch.sum(a_s_xt,1)

        #concat attended features
        feature_cat = torch.cat([feature1,feature2],1)

        return feature_cat, s_a, a_s

    def forward(self, graph, atom_feats, target, seq_lens, device):

        #process drug features with GraphConv layers
        h = self.relu(self.conv1(graph, atom_feats))
        h = self.relu(self.conv2(graph,h))
        h = self.relu(self.conv3(graph, h))

        #get batch number of nodes for each graph
        batch_num_objs = graph.batch_num_nodes()
        hs = self.graph_in_batch(batch_num_objs,h)

        #process protein features with embedding and Conv1d layers
        embedded_xt = self.embedding_xt(target)
        t = embedded_xt.permute(0,2,1)
        conv_xt = self.relu(self.conv_xt_1(t))
        conv_xt = self.relu(self.conv_xt_2(conv_xt))
        conv_xt = self.relu(self.conv_xt_3(conv_xt))

        #attention mechanism between drug and protein features
        att, s_a, a_s = self.attention_cal(hs,conv_xt, batch_num_objs, seq_lens,device  )

        #fully connected layers for affintiy prediction
        xc = self.dropout(self.relu(self.fc1(att)))
        xc = self.dropout(self.relu(self.fc2(xc)))
        out = self.out(xc)
        return out
    
def build_inference_input(smi_list, prot_seq:str):
    inpsz = len(smi_list)
    seq_len = len(prot_seq)

    # compound input
    comp_inp = [dgl.add_self_loop(smiles_to_bigraph(smi, node_featurizer=dl_data.atom_featurizer, 
                                                    edge_featurizer=dl_data.bond_featurizer)) 
                    for smi in smi_list]
    # protein input
    padded_seq = dl_data.seq_cat(prot_seq)
    prot_inp = np.tile(padded_seq, (inpsz,1)).tolist()
    # placeholder label - all zeros - collate() requires labels
    labels = np.zeros(inpsz).tolist()
    # protein lengths
    seq_lens = [seq_len]*inpsz

    # zipped input
    inp_data = list(zip(comp_inp, prot_inp, labels, seq_lens))
    return inp_data

def ensemble_inference(dl_models:List[DL_model], smi_list, prot_seq:str, bsz:int, dev):
    """ ensemble prediction by equal voting weights of m models """
    inp_data = build_inference_input(smi_list, prot_seq)
    dldr = data.DataLoader(inp_data, batch_size=bsz, shuffle=False, collate_fn=dl_data.collate)
    preds_col = []  # prediction batch collection
    for bat in dldr:
        drugs, targets, labels, seq_lens = bat  # labels are not used
        drugs = drugs.to(dev)
        atom_feats = drugs.ndata['h']
        targets = targets.to(dev)
        # labels = labels.to(dev)
        seq_lens = seq_lens.to(dev)

        n_by_one_preds = []
        for model in dl_models:
            output = model(drugs, atom_feats, targets, seq_lens, dev)  # (bsz, 1) tensor
            n_by_one_preds.append(output)
        ens_pred_batch = torch.hstack(n_by_one_preds)  # (bsz, m) tensor
        voted_batch = ens_pred_batch.mean(axis=1).detach().cpu()
        preds_col.append(voted_batch)
    return torch.cat(preds_col)
    