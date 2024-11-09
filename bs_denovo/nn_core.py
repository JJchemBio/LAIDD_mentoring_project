
from torch import nn, optim
import torch
from typing import List, Callable

class FFLayers(nn.Module):
    """
        You should specify at least one number for ff_sizes: List.
        The last number in ff_sizes will be used as final output size.
        Default activation is ReLU. Default dropout is 0.1. Batch norm is not used by default.
        The propagation order is Linear-> BN-> Act -> Dropout.
    """
    def __init__(self, inp_size, ff_sizes:List, act_func:Callable=None, drop_p=0.1, bnorm=False):
        super(FFLayers, self).__init__()
        if len(ff_sizes) < 1:
            raise ValueError("There should be at least one FF layer!")
        self.ff_sizes = ff_sizes
        self.ff_list = nn.ModuleList()
        self.drop_list = nn.ModuleList()
        self.bn_list = nn.ModuleList()
        if act_func is not None:
            self.act_func = act_func
        else:
            self.act_func = nn.functional.relu
        self.inp_size = inp_size
        self.drop_p = drop_p
        self.bnorm = bnorm
        
        # stacking the layers
        self.ff_list.append(nn.Linear(self.inp_size, self.ff_sizes[0]))
        for i in range(1, len(self.ff_sizes)):
            if self.bnorm == True:
                self.bn_list.append(nn.BatchNorm1d(num_features=self.ff_sizes[i-1]))
            self.drop_list.append(nn.Dropout(self.drop_p))
            self.ff_list.append(nn.Linear(self.ff_sizes[i-1], self.ff_sizes[i]))

    def forward(self, x):
        _x = x
        for i in range(len(self.ff_list)-1):
            ffout = self.ff_list[i](_x)
            if self.bnorm == True:
                ffout = self.bn_list[i](ffout)
            ffout = self.act_func(ffout)
            _x = self.drop_list[i](ffout)
        ffout = self.ff_list[-1](_x)
        return ffout # return the last feedforward output

class MultiLSTM(nn.Module):
    """ Implements a multi layer LSTM cell including an embedding layer
        and an output linear layer back to the size of the vocabulary """
    def __init__(self, emb_size, hidden_layer_units:List, voc_size):
        if len(hidden_layer_units) < 1:
            raise ValueError("There should be at least one hidden layer!")
        super(MultiLSTM, self).__init__()
        self.embedding = nn.Embedding(voc_size, emb_size)
        self.hidden_layer_units = hidden_layer_units
        self.num_hidden_layers = len(hidden_layer_units)
        self.lstm_list = nn.ModuleList()
        self.lstm_list.append(nn.LSTMCell(emb_size, hidden_layer_units[0]))
        for i in range(1, len(hidden_layer_units)):
            self.lstm_list.append(nn.LSTMCell(hidden_layer_units[i-1], hidden_layer_units[i]))
        self.linear = nn.Linear(hidden_layer_units[self.num_hidden_layers-1], voc_size)

        self.voc_size = voc_size
        self.emb_size = emb_size

    # forward() call performs only one step through the timeline.
    # Though, the process is done on the batch-wise.
    # x.shape = (batch_size) <- each example's t-th step token index
    # hs[i].shape = (batch_size, hl_units[i])
    def forward(self, x, hs:List, cs:List):
        emb_x = self.embedding(x)
        # emb_x.shape = (batch_size, feature_dim)
        hs[0], cs[0] = self.lstm_list[0](emb_x, (hs[0], cs[0]))
        for i in range(1, len(hs)):
            hs[i], cs[i] = self.lstm_list[i](hs[i-1], (hs[i], cs[i]))
        fc_out = self.linear(hs[len(hs)-1])
        return fc_out, hs, cs

class PositionalEncoding(nn.Module):
    """
        https://kaya-dev.tistory.com/8

        forward() returns matrix (batch_size, seq_len, d_model)
    """
    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()
        
        self.encoding = torch.zeros(max_len, d_model)
        self.encoding.requires_grad = False
        
        pos = torch.arange(max_len).float().unsqueeze(dim=1)  # (1 x max_len)
        _2i = torch.arange(0, d_model, step=2).float()  # index on embedding vector dim
        
        self.encoding[:,0::2] = torch.sin(pos / (10000**(_2i/d_model)))  # even emb dim index
        self.encoding[:,1::2] = torch.cos(pos / (10000**(_2i/d_model)))  # odd emb dim index
        
    def forward(self, x):
        """ x is expected to be a batch of encoded sequences (not embedded yet) with padding """
        batch_size, seq_len = x.size()
        return self.encoding[:seq_len,:].repeat((batch_size,1,1)).to(x.device)
    
class GPTModule(nn.Module):
    """
        Note that we are using batch_first=True option.
    """
    def __init__(self, voc_size, d_model, nhead, num_layers, max_len, dropout=0.1):
        super(GPTModule, self).__init__()
        self.voc_size = voc_size
        self.d_model = d_model
        self.embedding = nn.Embedding(voc_size, d_model)
        self.posenc = PositionalEncoding(d_model, max_len)
        self.max_len = max_len
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)  # batch first
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.linear = nn.Linear(d_model, voc_size)
    
    def forward(self, x, memory=None, tgt_mask=None, padding_mask=None):
        """
            x.shape = (batch_size, seq_len)
            memory.shape = (batch_size, source_seq_len, d_model)
            tgt_mask.shape = (seq_len, seq_len)
            padding_mask.shape = (batch_size, seq_len)
            padding_mask.dtype = torch.bool
            
            GPT don't need memory, but we leave it as option, since memory may be used as condition for seq generation.
        """
        bs, seq_len = x.shape
        emb_x = self.embedding(x)
        psen = self.posenc(x)
        _x = emb_x + psen

        if memory is None:
            memory = torch.zeros((bs, 1, self.d_model)).to(x.device)  # source_len = 1 for memory efficiency
        if tgt_mask is None:
            tgt_mask_float = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
            tgt_mask = torch.isinf(tgt_mask_float)
        if padding_mask is None:
            padding_mask = torch.zeros((bs, seq_len)).bool().to(x.device)  # consider there are no paddings

        _x = self.decoder(_x, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=padding_mask)
        return self.linear(_x)  # (batch_size, seq_len, voc_size)

class BERTModule(nn.Module):
    """
        Note that we are using batch_first=True option.
        Unlike GPT, BERTModule doesn't include a linear output at the end.
        The output of the forward() is the direct output from the transformer encoder.
    """
    def __init__(self, voc_size, d_model, nhead, num_layers, max_len, dropout=0.1):
        super(BERTModule, self).__init__()
        self.voc_size = voc_size
        self.d_model = d_model
        self.embedding = nn.Embedding(voc_size, d_model)
        self.posenc = PositionalEncoding(d_model, max_len)
        self.max_len = max_len
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)  # batch first
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, x, src_mask=None, padding_mask=None):
        """
            x.shape = (batch_size, seq_len)
            src_mask.shape = (seq_len, seq_len)
            padding_mask.shape = (batch_size, seq_len)
            padding_mask.dtype = torch.tool
                -> set True for padding positions, False for non-padding positions.
        """
        bs, seq_len = x.shape
        emb_x = self.embedding(x)
        psen = self.posenc(x)
        _x = emb_x + psen

        if src_mask is None:
            src_mask = torch.zeros((seq_len, seq_len)).bool().to(x.device)  # every position attends every position
        if padding_mask is None:
            padding_mask = torch.zeros((bs, seq_len)).bool().to(x.device)  # consider there are no paddings i.e. every position is enabled
        
        _x = self.encoder(_x, mask=src_mask, src_key_padding_mask=padding_mask)
        return _x  # (batch_size, seq_len, d_model)

class TransformerModule(nn.Module):
    """
        src and tgt used in forward() are both expected to have <BOS>, <EOS>, and <PAD>.
        Note that we are using batch_first=True option.
    """
    def __init__(self, d_model, nhead, src_vocsz, src_maxlen, num_enc_layers, 
                 tgt_vocsz, tgt_maxlen, num_dec_layers, dropout=0.1):
        super(TransformerModule, self).__init__()
        self.d_model = d_model
        
        self.src_vocsz = src_vocsz  # source vocabulary size
        self.src_maxlen = src_maxlen
        self.src_embedding = nn.Embedding(src_vocsz, d_model)
        self.src_posenc = PositionalEncoding(d_model, src_maxlen)

        self.tgt_vocsz = tgt_vocsz
        self.tgt_maxlen = tgt_maxlen
        self.tgt_embedding = nn.Embedding(tgt_vocsz, d_model)
        self.tgt_posenc = PositionalEncoding(d_model, tgt_maxlen)

        # batch_first = True
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_enc_layers,
                                          num_decoder_layers=num_dec_layers, dropout=dropout, batch_first=True)
        self.linear = nn.Linear(d_model, tgt_vocsz)
    
    def forward(self, src, tgt, tgt_mask=None, src_padding_mask=None, tgt_padding_mask=None):
        """
            src.shape = (batch_size, src_len), tgt.shape = (batch_size, tgt_len)
            tgt_mask.shape = (tgt_len, tgt_len)
            src_padding_mask.shape = (batch_size, src_len) bool type
            tgt_padding_mask.shape = (batch_size, tgt_len) bool type

            We don't use the triangular mask (autoregressive) for encoder part.
            If None is provided for padding_masks, we assume there are no paddings.
        """
        dev = src.device

        bs, src_len = src.shape
        bs2, tgt_len = tgt.shape
        if bs != bs2:
            raise ValueError("src and tgt batch sizes are different!")
        
        emb_src = self.src_embedding(src)
        src_psen = self.src_posenc(src)
        _src = emb_src + src_psen

        emb_tgt = self.tgt_embedding(tgt)
        tgt_psen = self.tgt_posenc(tgt)
        _tgt = emb_tgt + tgt_psen

        if tgt_mask is None:
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len).to(dev)
        if src_padding_mask is None:
            src_padding_mask = torch.zeros((bs, src_len)).bool().to(dev)  # consider there are no paddings
        if tgt_padding_mask is None:
            tgt_padding_mask = torch.zeros((bs, tgt_len)).bool().to(dev)

        # decoder_out.shape = (batch_size, tgt_len, d_model)        
        decoder_out = self.transformer(_src, _tgt, tgt_mask=tgt_mask, 
                         src_key_padding_mask=src_padding_mask, tgt_key_padding_mask=tgt_padding_mask)
        return self.linear(decoder_out)

class LRScheduler():
    """
        This class follows the function prototypes provided by torch.optim.lr_scheduler,
        but adding check_and_step() method that gives you more control.
    """
    def __init__(self, optimizer:optim.Optimizer):
        self.optimizer = optimizer
    def step():
        raise NotImplementedError()
    def check_and_step():
        raise NotImplementedError()

class ExponentialSchedule(LRScheduler):
    """
        Keep internal counter, and only updates when step_interval is met.
    """
    def __init__(self, optimizer, multiplier:float, step_interval:int):
        super(ExponentialSchedule, self).__init__(optimizer)
        self.multiplier = multiplier
        self.step_interval = step_interval
        self.counter = 0

    def step(self):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * self.multiplier
    
    def check_and_step(self):
        self.counter += 1
        if self.counter % self.step_interval == 0:
            self.step()
    
    def get_optimizer_lrs(self):
        lrs = []
        for pg in self.optimizer.param_groups:
            lrs.append(pg['lr'])
        return lrs