'''
Written by Jinhoon Jeong (Dong-A ST), as a participant of the LAIDD mentoring project
All scripts in bs_denovo were provided as part of the mentoring project and include some minor modifications.
This script is also mostly referred to the original script provided from the project.
'''
import pandas as pd
import numpy as np
import time, os
from bs_denovo import vocab
from bs_denovo.lang_data import StringDataset
from torch.utils.data import DataLoader
import torch
from torch import nn
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# base directory
base_dir = './'

# device
dev = 'cuda'

os.system('mkdir -p models')

# load data
df = pd.read_csv(f'{base_dir}/data/splitted_data_canonical_removed.tsv', sep='\t')
smiles_trn = df.loc[df['split']=='train', 'smiles'].tolist()
smiles_vld = df.loc[df['split']=='validation', 'smiles'].tolist()
smiles_tst = df.loc[df['split']=='test', 'smiles'].tolist()

# tokens
tokens = ['H', 'B', 'C', 'N', 'O', 'F', 'Na', 'Mg', 'Al', 'Si','P', 'S', 'Cl', 'Zn', 'As', 'Se', 'Br', 'Sr', 'Ag', 'Sn','Te', 'I', 'V', 'Li', 'Cr',
		  'c', 'n', 'o', 'b', 's', 'p', 'se', 'te', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0',
		  '-', '=', '#', '$', ':', '/', '\\', '+', '%', '[', ']', '(', ')', '@', '.',
		  '<CLS>', '<BOS>', '<EOS>', '<PAD>', '<MSK>', '<UNK>']

# vocabulary object (function: tokens, tok2id, id2tok, vocab_size, encode, decode, tokenize)
smiles_voc = vocab.SmilesVocabulary(list_tokens=tokens)

# make tensor dataset that transformed with tokenizing and encoding
dataset_trn = StringDataset(smiles_voc, smiles_trn)
dataset_vld = StringDataset(smiles_voc, smiles_vld)
dataset_tst = StringDataset(smiles_voc, smiles_tst)

# ENCODER
from bs_denovo.lang_lstm import EmbeddingLSTMConfig, EmbeddingLSTM
from bs_denovo.lang_vae import LangEncoderConfig #, LangEncoder
from bs_denovo.vae_lstm import LSTMEncoder

# encoder configuration
enc_conf = LangEncoderConfig(mem_sz=256, d_latent=128)

# embedding (within encoder)
emb_conf = EmbeddingLSTMConfig(device=dev, voc=smiles_voc, emb_size=128, hidden_layer_units=[256,256,256], ff_sizes=[256])
embedding = EmbeddingLSTM(emb_conf)

# encoder object
encoder = LSTMEncoder(enc_conf, embedding)

# batch data
dloader_vld = DataLoader(dataset_vld, batch_size=3, shuffle=True, collate_fn=dataset_vld.collate_fn)
for batch_vld in dloader_vld:
	break
'''
# get mu and logvar
mems = encoder.get_mem(batch_trn)
mu = encoder.z_means(mems)
logvar = encoder.z_var(mems)

# reparameterization
reparm = encoder.reparameterize(mu, logvar)
'''


# DECODER
#from bs_denovo.lang_vae import LangDecoder, LangDecoderConfig
from bs_denovo.vae_lstm import LSTMDecoder, LSTMDecoderConfig

# decoder configuration
dec_conf = LSTMDecoderConfig(device=dev, voc=smiles_voc, d_latent=128, emb_sz=128, hidden_layer_units=[256,256,256])

# decoder object
decoder = LSTMDecoder(dec_conf)


# VAE
from bs_denovo.lang_vae import LangVAE, LangVAEConfig

# VAE configuration
vae_conf = LangVAEConfig(bat_sz=1024, learn_rate=0.001, beta_list=[], max_len=150, DESCRIPTION='at 240910')
beta_list = [1.0 + (0.1*i) for i in range(200)]
vae_conf.beta_list = beta_list

# VAE object
VAE = LangVAE(vae_conf, encoder=encoder, decoder=decoder)

import logging
# logging set up
logging.basicConfig(filename="240910_vae_training.log", level=logging.INFO, filemode='w', format="%(asctime)-15s %(message)s", force=True)
logging.info("logger set up")


# functions related to validity, uniqueness, novelty, etc
from bs_denovo.gen_eval import standard_metrics

def log_standard_sample(vae_inst:LangVAE, ex_batch:torch.Tensor):
  log_str = ""
  sam_100, _ = vae_inst.randn_samples(100, greedy=False)
  stdmet = standard_metrics(sam_100, trn_set=[], subs_size=100)   # ignore novelty metric
  log_str += "--> Standard metrics(ignore novelty): \n" + str(stdmet) + '\n'
  # checking reconstruction of given examples
  _, mu, _, _ = vae_inst.encoder.forward_by_tgt(ex_batch)
  _, recon_seqs = vae_inst.decoder.decode2string(mu, greedy=True, max_len=vae_inst.conf.max_len)
  for ci, coup in enumerate(zip(ex_batch, recon_seqs)):
    log_str += "inp{}: {}\n".format(ci, coup[0].cpu().numpy())
    log_str += "out{}: {}\n".format(ci, coup[1].cpu().numpy())
  return log_str

import functools
log_add = functools.partial(log_standard_sample, vae_inst=VAE, ex_batch=batch_vld)

import multiprocessing
# model training
t1 = time.time()
logging.info("Training begins! {}".format(time.ctime()))
loss_collected = VAE.train(dataset_trn, epochs=200, save_period=10,
                                save_path="models/vae_e{}.ckpt", dl_njobs=8,
                                logging=logging, log_additional=log_add)
t2 = time.time()

logging.info("Training finished! {}".format(time.ctime()))
logging.info("recorded losses: \n{}".format(np.array(loss_collected)))
logging.info("Total training time: \n{} sec".format(round(t2-t1,2)))


