from bs_denovo.lang_vae import LangEncoder, LangEncoderConfig, LangDecoder, LangDecoderConfig
from bs_denovo.lang_lstm import EmbeddingLSTM
from bs_denovo.vocab import Vocabulary
from typing import List, Type
from dataclasses import dataclass, asdict
import torch
from torch import nn

class LSTMEncoder(LangEncoder):
	"""
		Simple recurrent encoder architecture.
		Make sure lstm_embr's final output dimension is the same as mem_sz.
	"""
	def __init__(self, conf:LangEncoderConfig, lstm_embr:EmbeddingLSTM):
		super(LSTMEncoder, self).__init__(conf)
		self.lstm_embr = lstm_embr

		# Bidirectional LSTM 적용
		self.bi_lstm = nn.LSTM(
			input_size=self.lstm_embr.config.ff_sizes[-1],
			hidden_size=conf.mem_sz // 2,  # Bidirectional이므로 출력의 절반만 hidden size로 설정
			num_layers=1,
			bidirectional=True,  # Bidirectional LSTM 활성화
			batch_first=True
		)

		self.bi_lstm.to(self.lstm_embr.device)

		# Note that we don't use optimizer in lstm_embr
		if conf.mem_sz != lstm_embr.config.ff_sizes[-1]:
			raise ValueError("mem_sz and last ff_sizes(of EmbeddingLSTM) are different!")
		self.z_means.to(lstm_embr.device)
		self.z_var.to(lstm_embr.device)

	def get_param_groups(self):
		param_groups = self.lstm_embr.get_param_groups()
		param_groups.append({"params":self.z_means.parameters()})
		param_groups.append({"params":self.z_var.parameters()})
		return param_groups
	
	def get_ckpt_dict(self):
		ckpt_dict = self.conf.dict()
		ckpt_dict['embr_dict'] = self.lstm_embr.get_ckpt_dict()
		ckpt_dict['z_means_state'] = self.z_means.state_dict()
		ckpt_dict['z_var_state'] = self.z_var.state_dict()
		ckpt_dict['bi_lstm_state'] = self.bi_lstm.state_dict()
		return ckpt_dict
	
	@staticmethod
	def construct_by_ckpt_dict(ckpt_dict, VocClass:Type[Vocabulary], dev='cpu'):
		# build embr
		embr_dict = ckpt_dict['embr_dict']
		lstm_embr = EmbeddingLSTM.construct_by_ckpt_dict(embr_dict, VocClass, dev)

		# build conf
		self_conf_dict = {}
		for k in LangEncoderConfig.__dataclass_fields__.keys():
			self_conf_dict[k] = ckpt_dict[k]
		self_conf = LangEncoderConfig(**self_conf_dict)

		# build self
		self_inst = LSTMEncoder(self_conf, lstm_embr)
		self_inst.z_means.load_state_dict(ckpt_dict['z_means_state'])
		self_inst.z_var.load_state_dict(ckpt_dict['z_var_state'])
		self_inst.bi_lstm.load_state_dict(ckpt_dict['bi_lstm_state'])
		return self_inst

	def get_mem(self, tgt_seqs:torch.Tensor, conds={}):
		"""
			conds['hs']=hidden_states, conds['cs']=cell_states <- can be used for conditional input.
			If conds is not provided, we will use None(no condition).
		"""
		if conds == {}:
			conds['hs'], conds['cs'] = None, None
		mem = self.lstm_embr.embed(tgt_seqs, init_hiddens=conds['hs'], init_cells=conds['cs'])
		mem, _ = self.bi_lstm(mem)
		return mem

@dataclass
class LSTMDecoderConfig(LangDecoderConfig):
	device:str
	hidden_layer_units: List

class LSTMDecoder(LangDecoder):
	"""
		Simple recurrent decoder architecture.
		Note that this decoder is not exactly a traditional Language model, 
			as its input sequence is in different form to the output.
			Because of this fact, it is not desirable to use the previous lstm implementations.
	"""
	def __init__(self, conf:LSTMDecoderConfig):
		super().__init__(conf)
		self.device = conf.device   
		self.embedding.to(self.device)
		self.z2inp.to(self.device)

		self.hl_units = conf.hidden_layer_units
		self.lstm_list = nn.ModuleList()
		self.lstm_list.append(nn.LSTMCell(self.deco_inp_sz, self.hl_units[0]))
		for i in range(1, len(self.hl_units)):
			self.lstm_list.append(nn.LSTMCell(self.hl_units[i-1], self.hl_units[i]))
		self.lstm_list.to(self.device)

		self.linear = nn.Linear(self.hl_units[len(self.hl_units)-1], self.voc.vocab_size)  # lstm output
		self.linear.to(self.device)

	def get_param_groups(self):
		param_groups = [{"params":self.lstm_list.parameters()}]
		param_groups.append({"params":self.linear.parameters()})
		return param_groups
	
	def get_ckpt_dict(self):
		ckpt_dict = self.conf.dict()
		ckpt_dict.pop('voc', None)
		ckpt_dict['voc_tokens'] = self.voc.tokens  # maintain the order of tokens

		ckpt_dict['embedding_state'] = self.embedding.state_dict()
		ckpt_dict['z2inp_state'] = self.z2inp.state_dict()
		ckpt_dict['lstm_list_state'] = self.lstm_list.state_dict()
		ckpt_dict['linear_state'] = self.linear.state_dict()
		return ckpt_dict
	
	@staticmethod
	def construct_by_ckpt_dict(ckpt_dict, VocClass:Type[Vocabulary], dev='cpu'):
		lang_tokens = ckpt_dict['voc_tokens']
		voc = VocClass(list_tokens=lang_tokens)

		# build conf
		self_conf_dict = {}
		for k in LSTMDecoderConfig.__dataclass_fields__.keys():
			if k == 'voc':
				self_conf_dict['voc'] = voc
			else:
				self_conf_dict[k] = ckpt_dict[k]
		self_conf_dict['device'] = dev
		self_conf = LSTMDecoderConfig(**self_conf_dict)

		# build self
		self_inst = LSTMDecoder(self_conf)
		self_inst.embedding.load_state_dict(ckpt_dict['embedding_state'])
		self_inst.z2inp.load_state_dict(ckpt_dict['z2inp_state'])
		self_inst.lstm_list.load_state_dict(ckpt_dict['lstm_list_state'])
		self_inst.linear.load_state_dict(ckpt_dict['linear_state'])	
		return self_inst

	def forward_step(self, x, zvecs, hs:List, cs:List):
		"""
			forward_step() call performs only one step through the timeline.
			Though, the process is done on the batch-wise.
			x.shape = (bsz) <- each example's t-th step token index
			zvecs.shape = (bsz, d_latent)
			hs[i].shape = (bsz, hl_units[i])
		"""
		emb_x = self.embedding(x)  # emb_x.shape = (batch_size, emb_size)
		z_inp = self.z2inp(zvecs)  # z_inp.shape = (batch_size, emb_size)
		lstm_inp = torch.cat((emb_x, z_inp), dim=1)  # lstm_inp.shape = (bsz, emb_sz * 2)

		hs[0], cs[0] = self.lstm_list[0](lstm_inp, (hs[0], cs[0]))
		for i in range(1, len(hs)):
			hs[i], cs[i] = self.lstm_list[i](hs[i-1], (hs[i], cs[i]))
		fc_out = self.linear(hs[len(hs)-1])
		return fc_out, hs, cs
	
	def step_likelihood(self, xi, zvecs, hidden_states, cell_states):
		"""returns the prob and log_prob of xi(i-th tokens of a batch sequences) given states"""
		logits, hidden_states, cell_states = self.forward_step(xi, zvecs, hidden_states, cell_states)
		# logits.shape = (batch_size, vocab_size)
		log_prob = nn.functional.log_softmax(logits, dim=1)
		prob = nn.functional.softmax(logits, dim=1)
		return prob, log_prob, hidden_states, cell_states
 
	def forward_by_tgt_z(self, tgt_seqs:torch.Tensor, zvecs:torch.Tensor, conds={}):
		"""
			- Args:
				conds: conds['hs']=hidden_states, conds['cs']=cell_states can be provided for conditions.
					If conds={}, then we just use None for default behavior(no condition).
			- Returns:
				_prob_map: (bsz, slen, voc_sz) softmax output tensor of the given target batch.
					Note that we used (bsz, voc_sz, slen) tensor in mid-unrolling, but eventually transformed.					
		"""
		bsz, tlen = tgt_seqs.shape  # batch size and target length
		_tgt_seqs = tgt_seqs.to(self.device).long()

		hidden_states, cell_states = [], []
		if conds == {}:
			for i in range(len(self.hl_units)):
				hidden_states.append(_tgt_seqs.new_zeros(bsz, self.hl_units[i]).float())
				cell_states.append(_tgt_seqs.new_zeros(bsz, self.hl_units[i]).float())
		else:
			for i in range(len(self.hl_units)):
				hidden_states.append(conds['hs'][i])
				cell_states.append(conds['cs'][i])

		start_token = torch.full((bsz,1), self.bosi).long().to(self.device)  # BOS column
		x = torch.cat((start_token, _tgt_seqs[:, :-1]), dim=1)  # the last position of tgt_seqs won't be used for input.
		
		NLLLoss = _tgt_seqs.new_zeros(bsz).float() 
		likelihoods = _tgt_seqs.new_zeros(bsz, tlen).float()
		prob_map = _tgt_seqs.new_zeros((bsz, self.voc.vocab_size, tlen)).float()
		for step in range(tlen):
			# Note that we are sliding a vertical scanner (height=batch_size) moving on timeline.
			x_step = x[:, step]  ## (batch_size)

			# let's find x_t[i] where it is <PAD>. Only <PAD>s will be True.
			padding_where = (_tgt_seqs[:, step] == self.padi)
			non_paddings = ~padding_where

			prob, log_prob, hidden_states, cell_states = self.step_likelihood(x_step, zvecs, hidden_states, cell_states)
			prob_map[:, :, step] = prob

			# the output of the lstm should be compared to the ones at x_step+1 (=target_step)
			one_hot_labels = nn.functional.one_hot(_tgt_seqs[:, step], num_classes=self.voc.vocab_size)

			# one_hot_labels.shape = (batch_size, vocab_size)
			# Make all the <PAD> tokens as zero vectors.
			one_hot_labels = one_hot_labels * non_paddings.reshape(-1,1)

			likelihoods[:, step] = torch.sum(one_hot_labels * prob, 1)
			loss = one_hot_labels * log_prob
			loss_on_batch = -torch.sum(loss, 1) # this is the negative log loss
			NLLLoss += loss_on_batch

		_prob_map = prob_map.transpose(1, 2)  # _prob_map: (bsz, tlen, voc_sz)
		return _prob_map, likelihoods, NLLLoss

	def decode_z(self, zvecs:torch.Tensor, greedy=False, max_len=999, conds={}):
		bsz, _ = zvecs.shape  # batch size and target length
		_zvecs = zvecs.to(self.device)

		hidden_states, cell_states = [], []
		if conds == {}:
			for i in range(len(self.hl_units)):
				hidden_states.append(_zvecs.new_zeros(bsz, self.hl_units[i]).float())
				cell_states.append(_zvecs.new_zeros(bsz, self.hl_units[i]).float())
		else:
			for i in range(len(self.hl_units)):
				hidden_states.append(conds['hs'][i])
				cell_states.append(conds['cs'][i])

		x_step = torch.full((bsz,), self.bosi).long().to(self.device)  # BOS tokens at the first step
		
		sequences, prob_list = [], []
		prob_map = _zvecs.new_zeros((bsz, self.voc.vocab_size, max_len)).float()
		finished = torch.zeros(bsz).byte() # memorize if the example is finished or not.
		for step in range(max_len):
			prob, _, hidden_states, cell_states = self.step_likelihood(x_step, _zvecs, hidden_states, cell_states)
			## prob.shape = (bsz, vocab_size)
			prob_list.append(prob.view(bsz, 1, self.voc.vocab_size))

			if greedy == True:
				next_word = torch.argmax(prob, dim=1)
			else:
				next_word = torch.multinomial(prob, num_samples=1).view(-1)
			sequences.append(next_word.view(-1, 1))

			x_step = next_word.clone()  # next step input

			# is EOS sampled at a certain example?
			EOS_sampled = (next_word == self.eosi)
			finished = torch.ge(finished + EOS_sampled.cpu(), 1)
			# if all the examples have produced EOS once, we will break the loop
			if torch.prod(finished) == 1: break
		# Each element in sequences is in shape (bsz, 1)
		# concat on dim=1 to get (bsz, seq_len)
		bat_seqs = torch.cat(sequences, dim=1)
		# Each element in prob_list is in shape (bsz, 1, voc_sz)
		# concat on dim=1 to get (bsz, seq_len, voc_sz)
		prob_map = torch.cat(prob_list, dim=1)
		return bat_seqs, prob_map
	
