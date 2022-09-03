import torch
import pytorch_lightning as pl
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from rdkit import Chem

class RDMSMILESAE(pl.LightningModule):
    def __init__(
        self,
        latent_size,
        hidden_dim,
        num_layers,
        vocab,
        bidirectional=True,
        word_dropout=0.5,
        lr=0.001,
        kl_weight=0.3,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.hidden_factor = (2 if bidirectional else 1) * num_layers
        self.vocab = vocab

        self.embedding = nn.Embedding(vocab.vocab_size, hidden_dim, padding_idx=vocab.pad_idx)
        self.encoder = nn.GRU(
            hidden_dim,
            hidden_dim,
            batch_first=True,
            num_layers=num_layers,
            bidirectional=bidirectional,
        )
        self.decoder = nn.GRU(
            hidden_dim, hidden_dim, batch_first=True, num_layers=num_layers, bidirectional=False
        )
        self.output2latent = nn.Linear(hidden_dim * self.hidden_factor, latent_size)
        self.latent2hidden = nn.Linear(latent_size, hidden_dim * self.num_layers)
        self.outputs2vocab = nn.Linear(hidden_dim, vocab.vocab_size)
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=vocab.pad_idx, reduction='sum')
        self.word_dropout = word_dropout
        self.lr = lr
        self.kl_weight = kl_weight

        self.save_hyperparameters()

    def resize_hidden_encoder(self, h, batch_size):
        if self.bidirectional or self.num_layers > 1:

            h = h.view(batch_size, self.hidden_dim * self.hidden_factor)
        else:
            h = h.squeeze()
        return h

    def resize_hidden_decoder(self, h, batch_size):
        if self.num_layers > 1:
            # flatten hidden state
            h = h.view(self.num_layers, batch_size, self.hidden_dim)
        else:
            h = h.unsqueeze(0)
        return h

    # def mask_inputs(self, x):
    #     x_mutate = x.clone()
    #     prob = torch.randn_like(x.float())
    #     prob[(x_mutate - self.vocab.sos_idx) * (x_mutate - self.vocab.pad_idx) == 0] = 1
    #     x_mutate[prob < self.word_dropout] = self.vocab.unk_idx
    #     return x_mutate  

    def compute_loss(self, outputs, targets, lengths, mu, noise):
        targets = targets[:, : torch.max(torch.tensor(lengths)).item()].contiguous().view(-1)
        outputs = outputs.view(-1, outputs.size(2))
        r_loss = self.criterion(outputs, targets)
        return r_loss

    def enc_forward(self, x, lengths):
        batch_size = x.shape[0]
        e = self.embedding(x)
        x_packed = pack_padded_sequence(e, lengths, batch_first=True, enforce_sorted=False)
        _, h = self.encoder(x_packed)

        h = self.resize_hidden_encoder(h, batch_size)
        mu = self.output2latent(h)
        noise = torch.empty(mu.shape).normal(0, 0.1)
        z = torch.randn_like(z)
        z = mu + noise
        return z, mu, noise

    def dec_forward(self, x, z, lengths):
        batch_size = x.shape[0]
        h = self.latent2hidden(z)
        x = self.mask_inputs(x)
        e = self.embedding(x)

        packed_input = pack_padded_sequence(e, lengths, batch_first=True, enforce_sorted=False)
        h = self.resize_hidden_decoder(h, batch_size)
        outputs, _ = self.decoder(packed_input, h)
        padded_outputs = pad_packed_sequence(outputs, batch_first=True)[0]
        output_v = self.outputs2vocab(padded_outputs)
        return output_v

    def forward(self, data):
        with_bos, with_eos, lengths = data
        z, mu, noise = self.enc_forward(with_bos, lengths)
        outputs = self.dec_forward(with_bos, z, lengths)
        return outputs, with_eos, lengths, mu, noise

    def training_step(self, batch, batch_idx):
        outputs = self.forward(batch)
        r_loss = self.compute_loss(*outputs)
        r = {}
        loss = r_loss
        r['loss'] = loss
        for key in r:
            self.log(f'train_{key}', r[key])
        return r

    def validation_step(self, batch, batch_idx):
        outputs = self.forward(batch)
        r_loss = self.compute_loss(*outputs)
        r = {}
        loss = r_loss
        r['loss'] = loss
        for key in r:
            self.log(f'val_{key}', r[key])
        return r

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def create_model(self, hparams):
        return RDMSMILESAE(**hparams)

    def smiles_to_latent(self, smiles, canonicalize=False):
        if canonicalize:
            smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
        self.eval()

        seq = [self.vocab.char2idx[i] for i in smiles]
        seq = torch.tensor([seq]).long()
        e = self.embedding(seq)
        _, h = self.encoder(e)
        h = self.resize_hidden_encoder(h, 1)
        mu = self.output2latent(h)
        return mu.detach().numpy()

class RDMSMILESAECNN(pl.LightningModule):
    def __init__(
        self,
        latent_size,
        hidden_dim,
        de_num_layers,
        vocab,
        prop_output_size=2,
        word_dropout=0.5,
        lr=0.001,
        t_kl_weight=1.0,
        p_weight=0.5,
        c_step=20000,
        out_dim=128,
        disc_dim=64,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.de_num_layers = de_num_layers
        self.vocab = vocab
        self.num_classes = len(self.vocab.char2idx)
        self.encoder = nn.Sequential(
            nn.Conv1d(self.num_classes, 9, kernel_size=9),  # shape: batch, 9, 47;
            nn.BatchNorm1d(9),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Conv1d(9, 9, kernel_size=9),  # shape: batch, 9, 39
            nn.BatchNorm1d(9),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Conv1d(9, 10, kernel_size=11),  # shape: batch, 10, 29
            nn.BatchNorm1d(10),
            nn.Dropout(0.0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(
                390, out_dim
            ),  # 290 when max_len=54, 390 when max_len=64 ;depends on the vocab.max_len
            nn.BatchNorm1d(out_dim),
            nn.Dropout(0.2),
        )

        self.decoder = nn.GRU(
            self.num_classes + latent_size,
            hidden_dim,
            batch_first=True,
            dropout=0.2,
            num_layers=de_num_layers,
            bidirectional=False,
        )

        self.disc = nn.Sequential(
            nn.Linear(latent_size, disc_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(disc_dim, prop_output_size),
        )

        self.mu_fc = nn.Linear(out_dim, latent_size)
        self.logvar_fc = nn.Linear(out_dim, latent_size)

        self.latent2hidden = nn.Linear(latent_size, hidden_dim * self.de_num_layers)
        self.outputs2vocab = nn.Linear(hidden_dim, vocab.vocab_size)
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)
        self.prop_criterion = torch.nn.MSELoss()
        self.word_dropout = word_dropout
        self.lr = lr
        self.t_kl_weight = t_kl_weight
        self.kl_weight = 0