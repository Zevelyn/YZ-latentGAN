from numpy import np
import rdkit
import torch
from torch import nn
from tqdm import tqdm
from rdkit import Chem
# import selfies as sf

class Vocab:
    def __init__(self, df, smiles_col, char2idx=None, idx2char=None):
        self.sos_token = '<sos>'
        self.eos_token = '<eos>'
        self.pad_token = '<pad>'
        self.unk_token = '<unk>'
        self.char2idx = {}
        self.idx2char = {}
        self.df = df
        self.smiles_col = smiles_col
        self.special_tokens = [self.sos_token, self.eos_token, self.pad_token, self.unk_token]
        self.special_atom = {'Cl': 'Q', 'Br':'W', '[nH]': 'X', '[H]': 'Y'}
        self.extract_charset()

    def vocab_size(self):
        return len(self.char2idx)
    
    def extract_charset(self):
        print('extracting charset')
        i = 0
        for c in self.special_tokens:
            if c not in self.char2idx:
                self.char2idx[c] = i
                self.idx2char[i] = c
                i += 1
        all_smi = self.df[self.smiles_col].values
        lengths = []
        for smi in all_smi:
            smi = smi.replace('Cl', 'Q')
            smi = smi.replace('Br', 'W')
            smi = smi.replace('[nH]', 'X')
            smi = smi.replace('[H]', 'Y')

            lengths.append(len(smi))

            for c in smi:
                if c not in self.char2idx:
                    self.char2idx[c] = i
                    self.idx2char[i] = c
                    i += 1
        self.max_len = 64 ## size
    
class AEDataset:
    def __init__(self, df, smiles_col, vocab):
        self.df = df
        self.smiles_col = smiles_col
        self.vocab = vocab
        self.sos_token = self.vocab.sos_token
        self.eos_token = self.vocab.eos_token
        self.pad_token = self.vocab.pad_token
        self.unk_token = self.vocab.unk_token
        self.char2idx = self.vocab.char2idx
        self.idx2char = self.vocab.idx2char
        self.all_smiles = self.df[self.smiles_col].values
        self.max_len = vocab.max_len
        self.tokenize()
    
    def __getitem__(self, idx):
        src_seq = self.tokens[idx]
        length = len(src_seq)
        with_bos = torch.tensor([self.char2idx[self.sos_token]] + src_seq + [self.vocab_pad_idx] *(self.max_len - length), dtype=torch.long)
        with_eos = torch.tensor(src_seq + [self.char2idx[self.eos_token]] + [self.vocab_pad_idx] *(self.max_len - length), dtype=torch.long)
        return (with_bos, with_eos, length + 1)

    def __len__(self):
        return len(self.df)
    
    def collate(self, samples):
        with_bos, with_eos, lengths = list(zip(*samples))
        lengths = list(lengths)
        with_bos = torch.stack(with_bos, dim=0)
        with_eos = torch.stack(with_eos, dim=0)
        return with_bos, with_eos, lengths

    def get_vocab(self):
        return Vocab(self.char2idx, self.idx2char)

    def tokenize(self):
        print('tokenizing..')
        self.tokens = []
        all_smi = self.df[self.smiles_col].values
        for smi in all_smi:
            smi = smi.replace('Cl', 'Q')
            smi = smi.replace('Br', 'W')
            smi = smi.replace('[nH]', 'X')
            smi = smi.replace('[H]', 'Y')
            t = [self.char2idx[i] for i in smi]
            self.tokens.append(t)
        print('done..')
    
class RDMAEOheDataset:
    def __init__(self, df, smiles_col, vocab):
        self.df = df
        self.smiles_col = smiles_col
        self.vocab = vocab
        self.sos_token = self.vocab.sos_token
        self.eos_token = self.vocab.eos_token
        self.pad_token = self.vocab.pad_token
        self.unk_token = self.vocab.unk_token
        self.char2idx = vocab.char2idx
        self.idx2char = vocab.idx2char
        self.all_smiles = self.df[self.smiles_col].values
        self.max_len = self.vocab.max_len
        self.tokens = []

    def tokenize(self, smi):
        smi = smi.replace('Cl', 'Q')
        smi = smi.replace('Br', 'W')
        smi = smi.replace('[nH]', 'X')
        smi = smi.replace('[H]', 'Y')
        src = [self.char2idx[i] for i in smi]
        return src

    def enumerate(self, smi, enum=True):
        if enum:
            rsmi = Chem.MolToSmiles(Chem.MolFromSmiles(smi), doRandom=True)
            try:
                src = self.tokenize(rsmi)
            except KeyError:
                src = self.tokenize(smi)
            if len(src) > self.max_len:
                src = self.tokenize(smi)
        else:
            src = self.tokenize(smi)
        return src, len(src)

    def __getitem__(self, idx):
        smi_seq = self.all_smiles[idx]
        src_e, length_e = self.enumerate(smi_seq, enum=True)
        src_d, length_d = self.enumerate(smi_seq, enum=True)

        with_bos_en = torch.tensor(
            [self.char2idx[self.sos_token]]
            + src_e
            + [self.vocab.pad_idx] * (self.max_len - length_e),
            dtype=torch.long,
        )

        with_bos_de = torch.tensor(
            [self.char2idx[self.sos_token]]
            + src_d
            + [self.vocab.pad_idx] * (self.max_len - length_d),
            dtype=torch.long,
        )

        with_eos = torch.tensor(
            src_d
            + [self.char2idx[self.eos_token]]
            + [self.vocab.pad_idx] * (self.max_len - length_d),
            dtype=torch.long,
        )
        return (with_bos_en, with_bos_de, with_eos, length_d + 1)

    def __len__(self):
        return len(self.df)

    def collate(self, samples):
        with_bos_en, with_bos_de, with_eos, lengths = list(zip(*samples))
        lengths = list(lengths)
        with_bos_en = torch.stack(with_bos_en, dim=0)
        with_bos_de = torch.stack(with_bos_de, dim=0)
        with_eos = torch.stack(with_eos, dim=0)
        return with_bos_en, with_bos_de, with_eos, lengths

    def get_vocab(self):
        return Vocab(self.char2idx, self.idx2char)
