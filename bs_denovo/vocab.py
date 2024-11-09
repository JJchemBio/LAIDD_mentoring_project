import numpy as np

class Vocabulary(object):
    
    special_tokens = ['<CLS>','<BOS>','<EOS>','<PAD>','<MSK>','<UNK>']
    def get_clsi(self): return self.tok2id['<CLS>']
    def get_bosi(self): return self.tok2id['<BOS>']
    def get_eosi(self): return self.tok2id['<EOS>']
    def get_padi(self): return self.tok2id['<PAD>']
    def get_mski(self): return self.tok2id['<MSK>']
    def get_unki(self): return self.tok2id['<UNK>']

    def __init__(self, list_tokens=None, file_name=None):
        """
            If file doesn't contain one of the special tokens, 
            we manually add the token to the end of self.tokens.
        """
        if list_tokens is None and file_name is None:
            print("Please specify list_tokens or file_name !!")
            return
        if file_name is not None:
            with open(file_name, 'r') as f:
                list_tokens = [line.strip() for line in f.readlines()]
        for spc in self.special_tokens:
            if spc not in list_tokens:
                list_tokens.append(spc)
        self.init_vocab(list_tokens)

    def init_vocab(self, list_tokens):
        self.tokens = list_tokens
        self.vocab_size = len(self.tokens)
        self.tok2id = dict(zip(self.tokens, range(self.vocab_size)))
        self.id2tok = {v: k for k, v in self.tok2id.items()}

    def have_invalid_token(self, token_list):
        for i, token in enumerate(token_list):
            if token not in self.tok2id.keys():
                return True
        return False

    def encode(self, token_list):
        """Takes a list of tokens (eg ['C','(','Br',')']) and encodes to array of indices"""
        if type(token_list) != list:
            print("encode(): the input was not a list type!!!")
            return None
        idlist = np.zeros(len(token_list), dtype=np.int32)
        for i, token in enumerate(token_list):
            try:
                idlist[i] = self.tok2id[token]
            except KeyError as err:
                print("encode(): KeyError occurred! %s"%err)
                raise
        return idlist
        
    def decode(self, idlist):
        """Takes an array of indices and returns the corresponding list of tokens"""
        return [self.id2tok[i] for i in idlist]

    def truncate_eos(self, batch_seqs:np.ndarray):
        """
            This function cuts off the tokens(id form) after the first <EOS> in each sample of batch.
            - Input: batch of token lists np.ndarray(batch_size x seq_len)
            - Output: truncated sequence list
        """
        bs, _ = batch_seqs.shape
        seq_list = []
        for i in range(bs):
            ids = batch_seqs[i].tolist()
            # append EOS at the end
            ids.append(self.get_eosi())
            # find EOS position of first encounter
            EOS_pos = ids.index(self.get_eosi())
            # get the seq until right before EOS
            seq_list.append(ids[0:EOS_pos])
        return seq_list

    def locate_specials(self, seq):
        """Return special (BOS, EOS, PAD, or any custom special) positions in the token id sequence"""
        spinds = [self.tok2id[spt] for spt in self.special_tokens]
        special_pos = []
        for i, token in enumerate(seq):
            if token in spinds:
                special_pos.append(i)
        return special_pos

    def tokenize(self, string):
        """Implement tokenization of string->List(token)"""
        raise NotImplementedError("Vocabulary.tokenize() needs to be implemented!!")

class SmilesVocabulary(Vocabulary):
    def __init__(self, list_tokens=None, file_name=None):
        super(SmilesVocabulary, self).__init__(list_tokens, file_name)
        self.multi_chars = set()
        for token in self.tokens:
            if len(token) >= 2 and token not in self.special_tokens:
                self.multi_chars.add(token)
    
    def tokenize(self, string):
        """
            Tokenization of string->List(token).
            Note that we expect "string" not to contain any special tokens.
        """
        # start with spliting with multi-char tokens
        token_list = [string] 
        for k_token in self.multi_chars:
            new_tl = []
            for elem in token_list:
                sub_list = []
                # split the sub smiles with the multi-char token
                splits = elem.split(k_token)
                # sub_list will have multi-char token between each split
                for i in range(len(splits) - 1):
                    sub_list.append(splits[i])
                    sub_list.append(k_token)
                sub_list.append(splits[-1]) 
                new_tl.extend(sub_list)
            token_list = new_tl
    
        # Now, only one-char tokens to be parsed remain.
        new_tl = []
        for token in token_list:
            if token not in self.multi_chars:
                new_tl.extend(list(token))
            else:
                new_tl.append(token)
        # Note that invalid smiles characters can be produced, if the smiles contains un-registered characters.
        return new_tl

class SingleCharacterVocabulary(Vocabulary):
    def __init__(self, list_tokens=None, file_name=None):
        super(SingleCharacterVocabulary, self).__init__(list_tokens, file_name)
        for tok in self.tokens:
            if tok not in self.special_tokens and len(tok) != 1:
                raise ValueError("initialized SingleCharacterVocabulary with some tokens with multiple characters!!")
    
    def tokenize(self, string):
        """
            Tokenization of string->List(token).
            Note that we expect "string" not to contain any special tokens.
        """
        return list(string)
    