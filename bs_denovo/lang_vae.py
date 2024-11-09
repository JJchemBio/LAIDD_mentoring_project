"""
    Implementation was based on:
        https://github.com/oriondollar/TransVAE

    However, note that this implement hugely deviated from the original ones.
"""
from dataclasses import dataclass, asdict
from bs_denovo.vocab import Vocabulary
from bs_denovo.lang_data import StringDataset
# from bs_denovo.gen_eval import standard_metrics
import torch
from torch import nn, optim
from torch.utils import data
from typing import List, Type
import numpy as np

@dataclass
class LangEncoderConfig:
    mem_sz:int
    d_latent:int
    def dict(self):
        return {k: v for k, v in asdict(self).items()}
    @classmethod
    def from_instance(cls, instance):
        return cls(**asdict(instance))
    
class LangEncoder:
    def __init__(self, conf:LangEncoderConfig):
        self.conf = conf
        self.z_means = nn.Linear(conf.mem_sz, conf.d_latent)
        self.z_var = nn.Linear(conf.mem_sz, conf.d_latent)

    def get_param_groups(self):
        raise NotImplementedError("Please implement this function by inheritance.")
    def get_ckpt_dict(self):
        raise NotImplementedError("Please implement this function by inheritance.")
    @staticmethod
    def construct_by_ckpt_dict(ckpt_dict, VocClass:Type[Vocabulary], dev='cpu'):
        raise NotImplementedError("Please implement this function by inheritance.")
    
    def get_mem(self, tgt_seqs:torch.Tensor, conds={}):
        """
            Get mem vectors with target sequences(tgt_seqs:(bsz, tlen)).
            returned mem is the vectors before getting to mu and var.
        """
        raise NotImplementedError("Please implement this function by inheritance.")

    def forward_by_tgt(self, tgt_seqs:torch.Tensor, conds={}):
        mem = self.get_mem(tgt_seqs, conds)  # mem is the vectors before getting to mu and var
        mu, logvar = self.z_means(mem), self.z_var(mem)
        repar = self.reparameterize(mu, logvar)
        return repar, mu, logvar, mem
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

@dataclass
class LangDecoderConfig:
    voc:Vocabulary  # voc is explicitly stated in decoder conf, as it needs sequence generation functionality.
    emb_sz:int  # decoder input size = (emb_sz * 2) at each position
    d_latent:int  # z vector dimension
    def dict(self):
        return {k: v for k, v in asdict(self).items()}
    @classmethod
    def from_instance(cls, instance):
        return cls(**asdict(instance))

class LangDecoder:
    def __init__(self, conf:LangDecoderConfig):
        self.conf = conf
        self.voc = conf.voc
        self.deco_inp_sz = conf.emb_sz * 2

        self.bosi = self.voc.get_bosi()
        self.eosi = self.voc.get_eosi()
        self.padi = self.voc.get_padi()

        self.embedding = nn.Embedding(self.voc.vocab_size, conf.emb_sz)  # token embedding
        self.z2inp = nn.Linear(conf.d_latent, conf.emb_sz)  # z vector to input

    def get_param_groups(self):
        raise NotImplementedError("Please implement this function by inheritance.")
    def get_ckpt_dict(self):
        raise NotImplementedError("Please implement this function by inheritance.")
    @staticmethod
    def construct_by_ckpt_dict(ckpt_dict, VocClass:Type[Vocabulary], dev='cpu'):
        raise NotImplementedError("Please implement this function by inheritance.")    
    
    def forward_by_tgt_z(self, tgt_seqs:torch.Tensor, zvecs:torch.Tensor, conds={}):
        """
            For given target sequences and z vectors of a batch, 
                return softmax output, probabilities for the target tokens, NLL of the given sequence.
            This function uses teachers forcing method for losses.
            Make sure zvecs is in self.device.
            - Args:
                tgt_seqs: (batch_size, seq_len) A batch of sequences in integer. 
                    <EOS> and <PAD> should be already in. <BOS> is not in.
                zvecs: (batch_size, d_latent) A batch of latent z vectors. In training, it is a reparameterized vector
                    from mu and logvar. In random sampling, it is a sampled latent from pre-defined prior.
            - Outputs:
                prob_map: (bsz, slen, voc_sz) softmax output tensor of the given target batch
                likelihoods: (batch_size, seq_length) likelihood for each position at each example
                NLLosses: (batch_size) negative log likelihood for each example
        """
        raise NotImplementedError("Please implement this function by inheritance.")
    
    def decode_z(self, zvecs:torch.Tensor, greedy=False, max_len=999, conds={}):
        """
            For given z vectors of a batch, returns:
                1. decoded output sequence (bsz, slen) - junk tokens could be sampled after EOS.
                2. probability(softmax) output at each position (bsz, slen, voc_sz)
            When greedy=False, multinomial sampling is performed at the language model output.
            When greedy=True, output tokens with max probability value is selected.
        """
        raise NotImplementedError("Please implement this function by inheritance.")
    
    def decode2string(self, zvecs:torch.Tensor, greedy=False, max_len=999, conds={}):
        """
            For given z vectors of a batch, returns:
                1. output string translated by vocab - sequences are truncated after EOS.
                2. decoded output sequence (bsz, slen) - junk tokens could be sampled after EOS.
            When greedy=False, multinomial sampling is performed at the language model output.
            When greedy=True, output tokens with max probability value is selected.
        """
        seqs, _ = self.decode_z(zvecs, greedy, max_len, conds)
        seqs_np = seqs.cpu().numpy()
        seqs_trcd = self.voc.truncate_eos(seqs_np)  # seqs truncated at EOS
        strings = ["".join(self.voc.decode(tok)) for tok in seqs_trcd]
        return strings, seqs

@dataclass
class LangVAEConfig:
    bat_sz:int  # batch size
    learn_rate:float 
    beta_list:List  # list of beta values to be used in training progress
    max_len: int  # maximum sequence length
    DESCRIPTION: str
    def dict(self):
        return {k: v for k, v in asdict(self).items()}
    @classmethod
    def from_instance(cls, instance):
        return cls(**asdict(instance))

class LangVAE:
    """
        VAE for sequence generation.
    """
    def __init__(self, conf:LangVAEConfig, encoder:LangEncoder, decoder:LangDecoder):
        self.conf = conf
        self.encoder = encoder
        self.decoder = decoder
        self.opt = optim.Adam(self.get_param_groups(), lr=conf.learn_rate)
        self.prog_num = 0
        # prog_num has several uses. 1) fill save path placeholder, 
        # 2) selecting beta value in beta_list by prog_num as index.

    def get_param_groups(self):
        param_groups = self.encoder.get_param_groups()
        param_groups.extend(self.decoder.get_param_groups())
        return param_groups
            
    def save_model(self, saveto):
        print("model is being saved to: ", saveto)
        ckpt_dict = self.conf.dict()
        ckpt_dict['encoder_dict'] = self.encoder.get_ckpt_dict()
        ckpt_dict['decoder_dict'] = self.decoder.get_ckpt_dict()
        ckpt_dict['opt_state'] = self.opt.state_dict()
        ckpt_dict['prog_num'] = self.prog_num
        torch.save(ckpt_dict, saveto)
    
    @staticmethod
    def construct_by_ckpt_dict(ckpt_dict, VocClass:Type[Vocabulary], EncClass:Type[LangEncoder],
                               DecClass:Type[LangDecoder], dev='cpu'):
        # build encoder
        encoder_dict = ckpt_dict['encoder_dict']
        encoder = EncClass.construct_by_ckpt_dict(encoder_dict, VocClass, dev)

        # build decoder
        decoder_dict = ckpt_dict['decoder_dict']
        decoder = DecClass.construct_by_ckpt_dict(decoder_dict, VocClass, dev)

        # build conf
        self_conf_dict = {}
        for k in LangVAEConfig.__dataclass_fields__.keys():
            self_conf_dict[k] = ckpt_dict[k]
        self_conf = LangVAEConfig(**self_conf_dict)

        # build self
        self_inst = LangVAE(self_conf, encoder, decoder)
        self_inst.opt.load_state_dict(ckpt_dict['opt_state'])
        self_inst.prog_num = ckpt_dict['prog_num']
        return self_inst

    def vae_loss(self, nllloss, mu, logvar, beta=1.0):
        """ 
            Binary Cross Entropy Loss + KL Divergence 
            - nllloss:(bsz,) is from forward_by_tgt_z of decoder. It is nll for each example in a batch.
            - mu:(bsz,d_latent) and logvar:(bsz,d_latent) are from forward_by_tgt of encoder
        """
        bce = torch.mean(nllloss)
        #### TODO: look out for the cases when kld becomes nan
        kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        if torch.any(torch.isnan(kld)):
            print("Nan value in kld detected!!!")
            print(kld)
        beta_kld = beta * kld
        return bce + beta_kld, bce, kld
        
    def train(self, seq_ds:StringDataset, epochs:int, save_period:int, save_path:str, dl_njobs=1,
              logging=None, log_additional=None, debug=999999):
        """
            save_model() is called every save_period epochs.
            - save_path: this string should include one placeholder e.g. "./model/vae_ver1_e{}.ckpt"
        """
        bsz = self.conf.bat_sz

        dldr = data.DataLoader(seq_ds, batch_size=bsz, shuffle=True, collate_fn=seq_ds.collate_fn, num_workers=dl_njobs)
        loss_collect = []  # element: [loss, bce, kld]
        for epo in range(1, epochs+1):
            if logging is not None:
                logging.info("--- epoch {}".format(epo))

            cur_beta = self.conf.beta_list[self.prog_num]
            losses, bces, klds = [], [], []
            for bi, bat_seqs in enumerate(dldr):
                repar, mu, logvar, mem = self.encoder.forward_by_tgt(bat_seqs)
                pmap, likelihoods, NLLLoss = self.decoder.forward_by_tgt_z(bat_seqs, repar)
                loss, bce, kld = self.vae_loss(NLLLoss, mu, logvar, beta=cur_beta)

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                
                losses.append(loss.item())
                bces.append(bce.item())
                klds.append(kld.item())
                if (logging is not None) and ((bi+1)%debug == 0):
                    logging.info("-- step {} - bce {} - kld {}".format(bi, bce, kld))
            
            # log after finishing an epoch
            self.prog_num += 1
            epo_loss = np.round([np.mean(losses), np.mean(bces), np.mean(klds)], decimals=4)
            loss_collect.append(epo_loss)
            if logging is not None:
                logging.info("--> prog_num {} - epoch loss:{}, bce:{}, kld:{} - used beta:{}".format(
                    self.prog_num, epo_loss[0], epo_loss[1], epo_loss[2], cur_beta
                ))
                if log_additional is not None:
                    logging.info(log_additional())
            
            # save in specified period
            if epo%save_period == 0:
                self.save_model(save_path.format(self.prog_num))
            
        return loss_collect
    
    def reconstruct(self, seq_ds:StringDataset, dl_njobs=1):
        """
            Make sure the provided input sequences don't exceed conf.max_len.
            Returns - 1. reconstructed strings, 2. input seq batches, 3. output seq batches
        """
        bsz = self.conf.bat_sz

        dldr = data.DataLoader(seq_ds, batch_size=bsz, shuffle=False, collate_fn=seq_ds.collate_fn, num_workers=dl_njobs)
        recon_strs, inp_bats, recon_bats  = [], [], []  # batches are collected here
        for bi, bat_seqs in enumerate(dldr):
            inp_bats.append(bat_seqs)  # input batches
            repar, mu, logvar, mem = self.encoder.forward_by_tgt(bat_seqs)
            # reconstruction is done by zvec = mu, i.e. zero variance
            zvecs = mu
            strs, seqs = self.decoder.decode2string(zvecs, greedy=True, max_len=self.conf.max_len)
            recon_bats.append(seqs.cpu())  # output batches
            recon_strs.append(strs)  # reconstructed strings, item is a batch
        
        _recon_strs = []  # flatten the list of list
        for slist in recon_strs:
            _recon_strs += slist
        return _recon_strs, inp_bats, recon_bats
    
    def randn_samples(self, ssz, bsz=None, greedy=False):
        """
            Generates samples by z vectors from random normal.
            ssz: total sample size
            bsz: batch size
        """
        d_latent = self.encoder.conf.d_latent
        if bsz is None:
            bsz = self.conf.bat_sz

        sampled_bats, zvec_bats = [], []  # batches are collected
        cnt = 0
        while cnt < ssz:
            zvecs = torch.randn((bsz, d_latent))
            zvec_bats.append(zvecs)
            gen_strs, _ = self.decoder.decode2string(zvecs, greedy, self.conf.max_len)
            sampled_bats.append(gen_strs)
            cnt += bsz

        samples = []  # flatten the list of list
        for slist in sampled_bats:
            samples += slist
        sam_zvecs = torch.vstack(zvec_bats)
        return samples[:ssz], sam_zvecs[:ssz]
