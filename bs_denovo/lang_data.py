"""
    If you want to use following Dataset with DataLoader, please specify collate_fn option 
        when you initialize DataLoader. We coded proper collate_fn to be used in
        each Dataset class.
"""

from bs_denovo.vocab import Vocabulary
from torch.utils import data
from torch.nn.utils import rnn
import torch
from typing import List, Callable

class VectorDataset(data.Dataset):
    def __init__(self, vecs):
        self.vecs = vecs
    
    def __len__(self):
        return len(self.vecs)
    
    def __getitem__(self, idx):
        return torch.tensor(self.vecs[idx])
    
    def collate_fn(self, batch):
        return torch.stack(batch)

class StringDataset(data.Dataset):
    """
        This is initialized with vocab, and list of strings.
        The strings will be tokenized and encoded by the vocab when __getitem__ is called.
        As tokenizing, we add EOS token at the end of the sequence.
    """
    def __init__(self, voc:Vocabulary, strings):
        self.voc = voc
        self.strings = strings
    
    def __len__(self):
        return len(self.strings)
    
    def __getitem__(self, idx):
        toks = self.voc.tokenize(self.strings[idx]) + [self.voc.id2tok[self.voc.get_eosi()]]
        return torch.tensor(self.voc.encode(toks))

    def collate_fn(self, batch):
        string_ids_tensor = rnn.pad_sequence(batch, batch_first=True, 
                                            padding_value=self.voc.get_padi())
        return string_ids_tensor
    
class String2FeatDataset(data.Dataset):
    """
        This is initialized with vocab, list of strings, and featurization function.
        __getitem__ will call the featurization for a string of index, and return the feature vector.
    """
    def __init__(self, voc:Vocabulary, strings, featzr:Callable):
        self.voc = voc
        self.strings = strings
        self.featzr = featzr
    
    def __len__(self):
        return len(self.strings)
    
    def __getitem__(self, idx):
        feat_vec = self.featzr(self.strings[idx])
        return torch.tensor(feat_vec)
    
    def collate_fn(self, batch):
        return torch.vstack(batch)

class StringLabelDataset(data.Dataset):
    """
        This is initialized with vocab, list of strings, and list of labels.
        The strings will be tokenized and encoded by the vocab when __getitem__ is called.
        As tokenizing, we add EOS token at the end of the sequence.

        Make sure each element of labels is 1-d array, that is, if the label is just a single value,
        wrap the value with np.array([value]). In other words, pass "labels=value_list.reshape(-1,1)".
        For LSTMClassifier, you need to put one_hot encoding as labels.
    """
    def __init__(self, voc:Vocabulary, strings, labels):
        if len(strings) != len(labels):
            raise ValueError("Please make sure the lengths of string list and label list are same!!")
        self.voc = voc
        self.strings = strings
        self.labels = labels
    
    def __len__(self):
        return len(self.strings)
    
    def __getitem__(self, idx):
        toks = self.voc.tokenize(self.strings[idx]) + [self.voc.id2tok[self.voc.get_eosi()]]
        return torch.tensor(self.voc.encode(toks)), torch.tensor(self.labels[idx])

    def collate_fn(self, batch):
        string_ids_list, label_list = [],[]
        for string_ids, label in batch:
            string_ids_list.append(string_ids)
            label_list.append(label)
        label_tensor = torch.vstack(label_list)
        string_ids_tensor = rnn.pad_sequence(string_ids_list, batch_first=True, 
                                            padding_value=self.voc.get_padi())
        return string_ids_tensor, label_tensor

class String2StringDataset(data.Dataset):
    """
        This is initialized with two vocabs, and lists of input strings and output strings.
        Typical use case of this dataset is the machine translation modeling.
        The strings will be tokenized and encoded by the vocabs when __getitem__ is called.
        As tokenizing, we add EOS token at the end of the sequence.
    """
    def __init__(self, in_voc:Vocabulary, in_strs, out_voc:Vocabulary, out_strs):
        if len(in_strs) != len(out_strs):
            raise ValueError("Please make sure the lengths of the two string lists are the same!!")
        self.in_voc = in_voc
        self.in_strs = in_strs
        self.out_voc = out_voc
        self.out_strs = out_strs
        
    def __len__(self):
        return len(self.in_strs)
    
class SeqMasker:
    """
        Make sure every seqence has <EOS> token. 
        We regard tokens coming after first <EOS> are all <PAD>.
    """
    def __init__(self, voc:Vocabulary, device='cpu'):
        self.voc = voc
        self.device = device

    def find_first_eos(self, batch_seqs:torch.Tensor):
        """
            NOT SURE IF I WILL NEED THIS FUNC
            > returns:
                - fe_poss: (bsz,) first EOS positions. If no EOS exist, it has slen+1 as value.
        """
        bsz, slen = batch_seqs.shape
        eosi = self.voc.get_eosi()
        row_col = torch.nonzero(batch_seqs==eosi).cpu().numpy()
        # row_col is in increasing order
        fe_poss = [slen+1 for k in range(bsz)]
        for _rc in row_col[::-1]:
            fe_poss[_rc[0]] = _rc[1]
        return fe_poss

    def pad_after_eos(self, batch_seqs:torch.Tensor):
        """ 
            Replacing tokens after EOS as PAD in batch. 
            Make sure all seqs in batch has at least one EOS.
            Note that returned tensor's length could be shorter than the input.
        """
        bsz, slen = batch_seqs.shape
        erows, ecols = torch.where(batch_seqs==self.voc.get_eosi())
        # e.g. [[C,c,EOS,Na],[c,EOS,1,EOS]] -> erows=[0,1,1], ecols=[2,1,3]

        trunc_seqs = [None]*bsz  # will collect truncated seqs here
        rbag = set(range(bsz))  # checking if certain row is already met
        for ri, ci in zip(erows.cpu().numpy(), ecols.cpu().numpy()):
            if ri in rbag:
                trunc_seqs[ri] = batch_seqs[ri,:ci+1]  # :ci+1 for including EOS
                rbag.remove(ri)

        if None in trunc_seqs:
            errmsg = "pad_after_eos() expects batch_seqs to have EOS in all its examples,"
            errmsg += "but " + str(rbag) + " examples don't have EOS!!"
            raise ValueError(errmsg)
        
        padded_seqs = rnn.pad_sequence(trunc_seqs, batch_first=True, 
                                       padding_value=self.voc.get_padi())
        return padded_seqs.to(self.device)

    def build_bert_mlm_input(self, batch_seqs:torch.Tensor, pad_mask:torch.Tensor, 
                             p_pos_pred=0.15, p_mask=0.8, p_rand=0.1, p_unch=0.1):
        """
            batch_seqs: (bs x seqlen) encoded tokens tensor
            pad_mask: (bs x seqlen) bool tensor specifying locations of padding tokens 
                -> True for padding positions, False for non-padding positions.
            p_pos_pred: probability of each position to be selected for prediction
            p_mask: probability of making the position's input to <MSK>
            p_rand: probability of converting it to random token
            p_unch: probability of unchanged.
            -- returns>
                bert_input: (bs x seqlen) encoded tokens tensor 
                pos_pred_bool: (bs x seqlen) whether BERT should predict the position (of batch_seqs) or not
        """
        vocsz = self.voc.vocab_size
        cls_ind = self.voc.get_clsi()
        bs, seqlen = batch_seqs.shape
        if (bs,seqlen) != pad_mask.shape:
            raise ValueError("shapes of batch_seqs and pad_mask are different!")
        if (p_mask + p_rand + p_unch) != 1.0:
            raise ValueError("p_mask + p_rand + p_unch should equal to 1.0 !!!")
        for seq in batch_seqs:
            if seq[0] != cls_ind:
                raise ValueError("seq in a batch should have <CLS> token at the beginning!")
        cls_mask = torch.zeros((bs,seqlen)).bool().to(self.device)
        cls_mask[:,0] = True

        # random number 0~1 sampling
        pos_pred_pmat = torch.rand((bs, seqlen)).to(self.device)

        # whether the position to be predicted or not
        pos_pred_bool = (pos_pred_pmat < p_pos_pred)
        # We don't predict padding positions
        pos_pred_bool = pos_pred_bool & (~pad_mask)
        # We don't predict CLS positions
        pos_pred_bool = pos_pred_bool & (~cls_mask)

        # rand num for input mask type selection 
        mask_type_pmat = torch.rand((bs, seqlen)).to(self.device)
        # <MSK> bool mat
        mask_bool = (mask_type_pmat < p_mask)
        # p_mask ~ (p_mask + p_rand) range is used for random token change
        rand_tok_bool = (mask_type_pmat >= p_mask) & (mask_type_pmat < (p_mask+p_rand))
        # others are unchanged
        unch_bool = (mask_type_pmat >= (p_mask+p_rand))

        # apply only for prediction positions
        mask_bool = mask_bool & pos_pred_bool
        rand_tok_bool = rand_tok_bool & pos_pred_bool
        unch_bool = unch_bool & pos_pred_bool  # not used
     
        # building MLM input
        bert_input = batch_seqs.clone()  # should it be detached ????
        bert_input = bert_input.masked_fill(mask_bool, self.voc.get_mski())
        bert_input[rand_tok_bool] = torch.randint(low=0, high=vocsz, size=(rand_tok_bool.sum(),)).to(self.device)

        return bert_input, pos_pred_bool

