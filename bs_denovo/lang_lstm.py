"""
    Note that this implementation contains some fixed components e.g. using ReLU, AdamOptimizer, ExponentialSchedule.
    You can manually change these components after constructing the objects.

    This implementation doesn't set the max seq length to be processed by LSTM. If you use a very long sequence,
    you probably run into memory explosion problem. This problem amplifies with a large batch size, so be careful.

    For floating point numbers in torch.Tensor, we always use Float type.
"""

from bs_denovo.vocab import Vocabulary
from bs_denovo.nn_core import MultiLSTM, FFLayers, ExponentialSchedule
from bs_denovo.lang_data import StringDataset, StringLabelDataset
import torch
from torch import nn, optim
from torch.utils import data
import numpy as np
from typing import List, Type
from dataclasses import dataclass, asdict

@dataclass
class LSTMConfig:
    device : str  # e.g."cuda" or "cpu"
    voc : Vocabulary
    emb_size : int = 128
    hidden_layer_units : List = None  # default for List is not allowed
    batch_size : int = 64
    init_lr : float = 0.001
    lr_mult : float = 1.0  ## default just maintains the lr
    lr_decay_interval : int = 1
    ckpt_path : str = "temp{}.ckpt"  # include one placeholder position
    
    def dict(self):
        return {k: v for k, v in asdict(self).items()}
    
    @classmethod
    def from_instance(cls, instance):
        return cls(**asdict(instance))
        ## https://stackoverflow.com/questions/54824893/python-dataclass-that-inherits-from-base-dataclass-how-do-i-upgrade-a-value-fr

class BaseLSTMWrapper():
    def __init__(self, config: LSTMConfig):
        self.config = config
        self.device = config.device
        self.lstm = MultiLSTM(config.emb_size, config.hidden_layer_units, config.voc.vocab_size)
        self.lstm.to(self.device)
        self.voc = config.voc
        self.bosi = self.voc.get_bosi()
        self.eosi = self.voc.get_eosi()
        self.padi = self.voc.get_padi()

        self.init_lr = config.init_lr
        self.lr_mult = config.lr_mult
        self.lr_decay_interval = config.lr_decay_interval
        self.batch_size = config.batch_size

        self.prog_num = 0  # number indicating the progress in training. used in saving ckpt name.
        self.ckpt_path = config.ckpt_path

        # optimizer initialization in superclass
        self.opt = optim.Adam(self.lstm.parameters(), lr=self.init_lr)
        # lr scheduler - you can replace this manually after initialization
        self.lr_schedule = ExponentialSchedule(self.opt, self.lr_mult, self.lr_decay_interval)

    def get_param_groups(self):
        return [{'params':self.lstm.parameters()}]

    def overwrite_emb(self, new_emb_mat: torch.Tensor, emb_grad=False):
        """
            Note that this function resets the optimizer, and the learning rate is set to init_lr.
        """
        if self.lstm.embedding.weight.shape != new_emb_mat.shape:
            raise ValueError("Embedding dimension not same!!")
        self.lstm.embedding = nn.Embedding.from_pretrained(new_emb_mat.clone())
        self.lstm.embedding.to(self.device)
        if emb_grad:
            self.lstm.embedding.weight.requires_grad = True  # default is False
        # reset the optimizer
        self.opt = optim.Adam(self.get_param_groups(), lr=self.init_lr)
        self.lr_schedule = ExponentialSchedule(self.opt, self.lr_mult, self.lr_decay_interval)

    def step_likelihood(self, xi, hidden_states, cell_states):
        """returns the prob and log_prob of xi given states"""
        # Note that we are not using x[:, step].view(-1,1) here.
        # If you look at the forward() of MultiLSTM, you see that it expects (batch_size).
        logits, hidden_states, cell_states = self.lstm(xi, hidden_states, cell_states)
        # logits.shape = (batch_size, vocab_size)
        log_prob = nn.functional.log_softmax(logits, dim=1)
        prob = nn.functional.softmax(logits, dim=1)
        return prob, log_prob, hidden_states, cell_states    

    def unroll_target(self, target, init_hiddens=None, init_cells=None):
        """
            For given sequences of a batch, 
                return softmax output, probabilities for the target tokens, NLL of the given sequence,
                hidden states, and cell states.
            - Args:
                target: (batch_size, seq_len) A batch of sequences in integer. 
                    <EOS> and <PAD> should be already in. <BOS> is not in.
                init_hiddens, init_cells: used for initializing hidden and cell states of stacked LSTMs.
            - Outputs:
                _prob_map: (bsz, slen, voc_sz) softmax output tensor of the given target batch
                    Note that we used (bsz, voc_sz, slen) tensor in mid-unrolling, but eventually transformed.
                likelihoods: (batch_size, seq_length) likelihood for each position at each example
                NLLosses: (batch_size) negative log likelihood for each example
                out_hs_list: [(batch_size, unit_size)] x num_layers
                out_cs_list: [(batch_size, unit_size)] x num_layers
        """
        target = torch.Tensor(target.float())
        target = target.to(self.device).long()
        # It is expected that all the seqs end with at least one EOS.
        # When making the input x, we will cut that last token at the end,
        # and add BOS token at the begining.
        batch_size, seq_length = target.size()
        start_token = target.new_zeros(batch_size, 1).long()
        start_token[:] = self.bosi # initialize with all BOS

        hl_units = self.lstm.hidden_layer_units
        hidden_states, cell_states = [], []
        out_hs_list, out_cs_list = [], []
        if init_hiddens is not None and init_cells is not None:
            for i in range(len(hl_units)):
                hidden_states.append(init_hiddens[i])
                cell_states.append(init_cells[i])
                out_hs_list.append(target.new_zeros(batch_size, hl_units[i]).float())
                out_cs_list.append(target.new_zeros(batch_size, hl_units[i]).float())
        else:
            for i in range(len(hl_units)):
                hidden_states.append(target.new_zeros(batch_size, hl_units[i]).float())
                cell_states.append(target.new_zeros(batch_size, hl_units[i]).float())
                out_hs_list.append(target.new_zeros(batch_size, hl_units[i]).float())
                out_cs_list.append(target.new_zeros(batch_size, hl_units[i]).float())

        x = torch.cat((start_token, target[:, :-1]), 1)
        NLLLoss = target.new_zeros(batch_size).float() 
        likelihoods = target.new_zeros(batch_size, seq_length).float()
        prob_map = target.new_zeros((batch_size, self.voc.vocab_size, seq_length)).float()
        for step in range(seq_length):
            # Note that we are sliding a vertical scanner (height=batch_size) moving on timeline.      
            x_step = x[:, step]  ## (batch_size)

            # let's find x_t[i] where it is <PAD>. Only <PAD>s will be True.
            # padding_where = (x_step == self.padi)
            padding_where = (target[:, step] == self.padi)  ### fixed: target is padding or not
            # padding_where.shape = (batch_size)
            non_paddings = ~padding_where
            non_padding_locs = torch.where(non_paddings)[0]  # index among batch where it is non-PAD

            prob, log_prob, hidden_states, cell_states = self.step_likelihood(x_step, hidden_states, cell_states)
            prob_map[:, :, step] = prob
            for i in range(len(hl_units)):
                # update output states only when the current token is non-PAD.
                # This is done so as for the early-ended examples should return the states right at the EOS,
                #    before they process to any further paddings.
                out_hs_list[i][non_padding_locs] = hidden_states[i][non_padding_locs]
                out_cs_list[i][non_padding_locs] = cell_states[i][non_padding_locs]
                
            # the output of the lstm should be compared to the ones at x_step+1 (=target_step)
            one_hot_labels = nn.functional.one_hot(target[:, step], num_classes=self.voc.vocab_size)

            # one_hot_labels.shape = (batch_size, vocab_size)
            # Make all the <PAD> tokens as zero vectors.
            one_hot_labels = one_hot_labels * non_paddings.reshape(-1,1)

            likelihoods[:, step] = torch.sum(one_hot_labels * prob, 1)
            loss = one_hot_labels * log_prob
            loss_on_batch = -torch.sum(loss, 1) # this is the negative log loss
            NLLLoss += loss_on_batch
        
        _prob_map = prob_map.transpose(1, 2)  # _prob_map: (bsz, slen, voc_sz)
        return _prob_map, likelihoods, NLLLoss, out_hs_list, out_cs_list

    def train():
        raise NotImplementedError("BaseLSTMWrapper.train() needs to be implemented!!")

    def get_ckpt_dict(self):
        ckpt_dict = self.config.dict()
        ckpt_dict.pop('voc', None)
        ckpt_dict['voc_tokens'] = self.voc.tokens  # maintain the order of tokens
        ckpt_dict['prog_num'] = self.prog_num
        ckpt_dict['lstm_state_dict'] = self.lstm.state_dict()
        ckpt_dict['opt_state_dict'] = self.opt.state_dict()  # params on the child class optimizer are also stored here
        return ckpt_dict

    def save(self):
        saveto = self.ckpt_path.format(str(self.prog_num))
        ckpt_dict = self.get_ckpt_dict()
        torch.save(ckpt_dict, saveto)

    def load_construct(device, voc:Vocabulary, load_ckpt_path, save_ckpt_path):
        """
            - Args:
                load_ckpt_path: path for loading ckpt
                save_ckpt_path: path with single placeholder(prog_num) for saving ckpt
        """
        raise NotImplementedError("BaseLSTMWrapper.load_construct() needs to be implemented!!")

@dataclass
class EmbeddingLSTMConfig(LSTMConfig):
    ff_sizes : List = None

class EmbeddingLSTM(BaseLSTMWrapper):
    """
        Add some feedforward layers on the hidden state outputs.
        Note that this doesn't state the task is classification or regression.
        That's for another class definition to follow.

        We only use ReLU as activation.
    """
    def __init__(self, config:EmbeddingLSTMConfig):
        super(EmbeddingLSTM, self).__init__(config)
        self.act_func = nn.functional.relu
        # We will concat stacked LSTM's all hidden states,
        # and use it for the feedforward input.
        self.conch_size = np.sum(self.lstm.hidden_layer_units)
        self.fflayers = FFLayers(self.conch_size, config.ff_sizes, self.act_func, drop_p=0)  ## use same relu for FF
        self.fflayers.to(self.device)

        # adding param_group to optimizer
        param_group = {'params':self.fflayers.parameters(), 'lr':self.init_lr}
        self.opt.add_param_group(param_group)

    def get_param_groups(self):
        return super(EmbeddingLSTM, self).get_param_groups() + [{'params':self.fflayers.parameters()}]
    
    def embed(self, target, init_hiddens=None, init_cells=None):
        _, _, _, hidden_states, cell_states = self.unroll_target(target, init_hiddens, init_cells)
        # concat the hidden states of stacks
        batch_input = torch.hstack(hidden_states)
        return self.fflayers(batch_input)

    def get_ckpt_dict(self):
        ckpt_dict = super(EmbeddingLSTM, self).get_ckpt_dict()
        ckpt_dict['ff_state_dict'] = self.fflayers.state_dict()
        return ckpt_dict
    
    @staticmethod
    def construct_by_ckpt_dict(ckpt_dict, VocClass:Type[Vocabulary], dev='cpu'):
        lang_tokens = ckpt_dict['voc_tokens']
        voc = VocClass(list_tokens=lang_tokens)

        # build conf
        self_conf_dict = {}
        for k in EmbeddingLSTMConfig.__dataclass_fields__.keys():
            if k == 'voc':
                self_conf_dict['voc'] = voc
            else:
                self_conf_dict[k] = ckpt_dict[k]
        self_conf_dict['device'] = dev
        self_conf = EmbeddingLSTMConfig(**self_conf_dict)

        # build self
        self_inst = EmbeddingLSTM(self_conf)
        self_inst.lstm.load_state_dict(ckpt_dict['lstm_state_dict'])
        self_inst.fflayers.load_state_dict(ckpt_dict['ff_state_dict'])
        self_inst.opt.load_state_dict(ckpt_dict['opt_state_dict'])
        self_inst.prog_num = ckpt_dict['prog_num']
        return self_inst

@dataclass
class LSTMPredictorConfig(EmbeddingLSTMConfig):
    out_dim : int = 1

class LSTMRegressor(EmbeddingLSTM):
    def __init__(self, config:LSTMPredictorConfig):
        super(LSTMRegressor, self).__init__(config)
        self.seqemb_size = self.fflayers.ff_sizes[-1]
        self.outreg = nn.Linear(self.seqemb_size, config.out_dim)
        self.outreg.to(self.device)

        param_group = {'params':self.outreg.parameters(), 'lr':self.init_lr}
        self.opt.add_param_group(param_group)

    def get_param_groups(self):
        return super(LSTMRegressor, self).get_param_groups() + [{'params':self.outreg.parameters()}]

    def regress(self, target, init_hiddens=None, init_cells=None):
        batch_seqemb = self.embed(target, init_hiddens, init_cells)
        return self.outreg(self.act_func(batch_seqemb))

    def train(self, dataset:StringLabelDataset, epochs:int, save_period:int, prog_save=True, 
              dl_njobs=1, debug=None):
        if prog_save: self.save()  # save initial state

        dldr = data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=dl_njobs,
                               collate_fn=dataset.collate_fn)
        epo_loss_list = []
        for epo in range(1, epochs+1):
            print("- epoch:", epo, " - progress:", self.prog_num)
            loss_collection = 0.0
            for bi, batch_data in enumerate(dldr):
                string_data = batch_data[0]
                predictions = self.regress(string_data)
                labels = batch_data[1].to(self.device).float()
                loss = nn.functional.mse_loss(labels, predictions)
                
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                loss_collection += loss.cpu().detach()
                if debug is not None:
                    if bi%debug == 0: print(loss.cpu().detach())

            epoch_loss = loss_collection / len(dldr)  # division by num of batches
            epo_loss_list.append(epoch_loss)
            print("-- loss:", epoch_loss)
            self.prog_num += 1  # single epoch training is done
            self.lr_schedule.check_and_step()
            if prog_save and (self.prog_num % save_period == 0):
                self.save()
        return epo_loss_list

    def get_ckpt_dict(self):
        ckpt_dict = super(LSTMRegressor, self).get_ckpt_dict()
        ckpt_dict['outreg_state_dict'] = self.outreg.state_dict()
        return ckpt_dict

    @staticmethod
    def load_construct(device, voc:Vocabulary, load_ckpt_path, save_ckpt_path):
        ckpt = torch.load(load_ckpt_path, map_location=device)
        ckpt['voc'] = voc

        config_dict = {k: ckpt[k] for k in LSTMPredictorConfig.__dataclass_fields__.keys()}
        config_dict['device'] = device
        config_dict['ckpt_path'] = save_ckpt_path

        conf = LSTMPredictorConfig(**config_dict)
        regsr = LSTMRegressor(conf)
        regsr.lstm.load_state_dict(ckpt['lstm_state_dict'])
        regsr.fflayers.load_state_dict(ckpt['ff_state_dict'])
        regsr.outreg.load_state_dict(ckpt['outreg_state_dict'])
        regsr.opt.load_state_dict(ckpt['opt_state_dict'])
        regsr.prog_num = ckpt['prog_num']
        return regsr

class LSTMClassifier(EmbeddingLSTM):
    def __init__(self, config:LSTMPredictorConfig):
        super(LSTMClassifier, self).__init__(config)
        self.seqemb_size = self.fflayers.ff_sizes[-1]
        self.outlogit = nn.Linear(self.seqemb_size, config.out_dim)
        self.outlogit.to(self.device)
        self.num_cls = config.out_dim  ## specific to classifier

        param_group = {'params':self.outlogit.parameters(), 'lr':self.init_lr}
        self.opt.add_param_group(param_group)

    def get_param_groups(self):
        return super(LSTMClassifier, self).get_param_groups() + [{'params':self.outlogit.parameters()}]

    def classify(self, target, init_hidden=None, init_cells=None):
        batch_seqemb = self.embed(target, init_hidden, init_cells)
        logit = self.outlogit(self.act_func(batch_seqemb))
        return nn.functional.softmax(logit, dim=1)

    def train(self, dataset:StringLabelDataset, epochs:int, save_period:int, prog_save=True, debug=None):
        if prog_save: self.save()  # save initial state

        dldr = data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True, collate_fn=dataset.collate_fn)
        for epo in range(1, epochs+1):
            print("- epoch:", epo, " - progress:", self.prog_num)
            loss_collection = 0.0
            epo_loss_list = []
            for bi, batch_data in enumerate(dldr):
                string_data = batch_data[0]
                predictions = self.classify(string_data)
                labels = batch_data[1].to(self.device).float()
                loss = nn.functional.binary_cross_entropy(predictions, labels)

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                loss_collection += loss.cpu().detach()
                if debug is not None:
                    if bi%debug == 0: print(loss.cpu().detach())
            
            epoch_loss = loss_collection / len(dldr)  # division by num of batches
            epo_loss_list.append(epoch_loss)
            print("-- loss:", epoch_loss)
            self.prog_num += 1
            self.lr_schedule.check_and_step()
            if prog_save and (self.prog_num % save_period == 0):
                self.save()
        return epo_loss_list

    def get_ckpt_dict(self):
        ckpt_dict = super(LSTMRegressor, self).get_ckpt_dict()
        ckpt_dict['outlogit_state_dict'] = self.outlogit.state_dict()
        return ckpt_dict

    @staticmethod
    def load_construct(device, voc:Vocabulary, load_ckpt_path, save_ckpt_path):
        ckpt = torch.load(load_ckpt_path, map_location=device)
        ckpt['voc'] = voc

        config_dict = {k: ckpt[k] for k in LSTMPredictorConfig.__dataclass_fields__.keys()}
        config_dict['device'] = device
        config_dict['ckpt_path'] = save_ckpt_path

        conf = LSTMPredictorConfig(**config_dict)
        clsfr = LSTMClassifier(conf)
        clsfr.lstm.load_state_dict(ckpt['lstm_state_dict'])
        clsfr.fflayers.load_state_dict(ckpt['ff_state_dict'])
        clsfr.outlogit.load_state_dict(ckpt['outlogit_state_dict'])
        clsfr.opt.load_state_dict(ckpt['opt_state_dict'])
        clsfr.prog_num = ckpt['prog_num']
        return clsfr

@dataclass
class LSTMGeneratorConfig(LSTMConfig):
    pass  ## just in case if something is needed in future

class LSTMGenerator(BaseLSTMWrapper):
    def __init__(self, config: LSTMGeneratorConfig):
        super(LSTMGenerator, self).__init__(config)
        ### No additional param_group to add in optimizer for Generator...

    def get_param_groups(self):
        return super(LSTMGenerator, self).get_param_groups()

    def sample_batch(self, max_len=100):
        """
            Sample a batch of sequences (token id format). Size is self.batch_size.
            The returning sequence is in rectangular shape.
            That is, some examples will contain junk tokens after the <EOS> positions.
            - Args:
                max_length: maximum length of the sequences
            - Outputs:
                seqs: (batch_size, seq_length) The sampled sequences.
                likelihoods: (batch_size, seq_len) likelihood for each sequence.
        """
        start_token = torch.zeros(self.batch_size).long().to(self.device)
        start_token[:] = self.bosi

        x = start_token.view(-1,1)  ## x.shape == (batch_size, 1)

        hl_units = self.lstm.hidden_layer_units
        hidden_states = []
        cell_states = []
        for i in range(len(hl_units)):
            hidden_state = x.new_zeros(self.batch_size, hl_units[i]).float() 
            hidden_states.append(hidden_state)
            cell_state = x.new_zeros(self.batch_size, hl_units[i]).float() 
            cell_states.append(cell_state)

        sequences = []
        likelihoods = x.new_zeros(self.batch_size, max_len).float() 
        finished = torch.zeros(self.batch_size).byte() # memorize if the example is finished or not.
        for step in range(max_len):
            prob, _, hidden_states, cell_states = self.step_likelihood(x.reshape(self.batch_size), hidden_states, cell_states)
            ## prob.shape = (batch_size, vocab_size)

            x = torch.multinomial(prob, num_samples=1).view(-1) ## x.shape = (batch_size)
            sequences.append(x.view(-1, 1))

            one_hot_labels = nn.functional.one_hot(x, self.voc.vocab_size) 
            ## one_hot_labels.shape == (batch_size, vocab_size)
            likelihoods[:, step] = torch.sum(one_hot_labels * prob, 1)

            x = x.data.clone()
            # is EOS sampled at a certain example?
            EOS_sampled = (x == self.eosi).data            
            finished = torch.ge(finished + EOS_sampled.cpu(), 1)
            # if all the examples have produced EOS once, we will break the loop
            if torch.prod(finished) == 1: break
        # Each element in sequences is in shape (batch_size x 1)
        # concat on dim=1 to get (batch_size x seq_len)
        sequences = torch.cat(sequences, 1)
        return sequences.data, likelihoods

    def sample_decode(self, ssize, max_len=100):
        """ 
            Sample some sequences and return the decoded ones (string format).
            The samples don't have EOS will be discarded.
        """
        generation = []
        while len(generation) <= ssize:
            tokens_list, _ = self.sample_batch(max_len)
            EOS_exist = [] # store which sample includes EOS token
            for i in range(self.batch_size):
                if self.eosi in tokens_list[i]:
                    EOS_exist.append(i)
            tokens_have_EOS = tokens_list[EOS_exist,:]

            # cut off after the first <EOS>
            trunc_seq_list = self.voc.truncate_eos(tokens_have_EOS.cpu().numpy())
            ##### cleaning part is gone...
            decoded_tokens = [self.voc.decode(seq) for seq in trunc_seq_list]
            seq_list = [''.join(tl) for tl in decoded_tokens]
            generation.extend(seq_list)
        return generation[:ssize]

    def train(self, dataset:StringDataset, epochs:int, save_period:int, prog_save=True, 
              dl_njobs=1, debug=None):
        if prog_save: self.save()  # save initial state

        dldr = data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=dl_njobs,
                               collate_fn=dataset.collate_fn)
        epo_loss_list = []
        for epo in range(1, epochs+1):
            print("- epoch:", epo, " - progress:", self.prog_num)
            loss_collection = 0.0
            for bi, batch_data in enumerate(dldr):
                _, likelihoods, NLLLoss, _, _ = self.unroll_target(batch_data)
                mean_loss = NLLLoss.mean()

                self.opt.zero_grad()
                mean_loss.backward()
                self.opt.step()
                loss_collection += mean_loss.cpu().detach()
                if debug is not None:
                    if bi%debug == 0: print(mean_loss.cpu().detach())
            
            epoch_loss = loss_collection / len(dldr)
            epo_loss_list.append(epoch_loss)
            print("-- loss:", epoch_loss)
            self.prog_num += 1
            self.lr_schedule.check_and_step()
            if prog_save and (self.prog_num % save_period == 0):
                self.save()
        return epo_loss_list

    def get_ckpt_dict(self):
        ckpt_dict = super(LSTMGenerator, self).get_ckpt_dict()
        return ckpt_dict

    @staticmethod
    def load_construct(device, voc:Vocabulary, load_ckpt_path, save_ckpt_path):
        ckpt = torch.load(load_ckpt_path, map_location=device)
        ckpt['voc'] = voc

        config_dict = {k: ckpt[k] for k in LSTMGeneratorConfig.__dataclass_fields__.keys()}
        config_dict['device'] = device
        config_dict['ckpt_path'] = save_ckpt_path

        conf = LSTMGeneratorConfig(**config_dict)
        generator = LSTMGenerator(conf)
        generator.lstm.load_state_dict(ckpt['lstm_state_dict'])
        generator.opt.load_state_dict(ckpt['opt_state_dict'])
        generator.prog_num = ckpt['prog_num']
        return generator
    
    @staticmethod
    def construct_by_ckpt_dict(ckpt_dict, voc:Vocabulary):
        """ 
            please check prog_num value after construct (caution for overwriting existing ckpt file)
        """
        ckpt_dict['voc'] = voc
        
        # build self
        self_conf_dict = {}
        for k in LSTMGeneratorConfig.__dataclass_fields__.keys():
            self_conf_dict[k] = ckpt_dict[k]
        self_conf = LSTMGeneratorConfig(**self_conf_dict)
        self_inst = LSTMGenerator(self_conf)
        self_inst.lstm.load_state_dict(ckpt_dict['lstm_state_dict'])
        self_inst.opt.load_state_dict(ckpt_dict['opt_state_dict'])
        self_inst.prog_num = ckpt_dict['prog_num']
        return self_inst
