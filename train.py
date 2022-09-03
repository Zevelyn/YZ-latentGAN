import torch
from sklearn import train_test_split
from _gan_dataset import AEDataset, Vocab
from _model import RDMSMILESAE
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

vocab = Vocab(df)

train_df, val_df = train_test_split(df, test_size=0.1, shuffle=True)
train = AEDataset(train_df, "SMILES", vocab=vocab)
val = AEDataset(val_df, "SMILES", vocab=vocab)
train_loader = torch.utils.data.DataLoader(train, batch_size = 128, shuffle=True, collate_fn=train.collate)
val_loader = torch.utils.data.DataLoader(val, batch_size = 128, shuffle=True, collate_fn=val.collate)

model = RDMSMILESAE( latent_size=256, hidden_dim=512, num_layers=3, vocab=vocab)
es = EarlyStopping(monitor='val_loss', patient=20, verbose=True)
logger = CSVLogger(save_dir='logs', name='ae')
checkpoint = ModelCheckpoint(monitor='val_loss', dirpath='ae', filename='ae-{epoch:02d}-{val_loss:.2f}', save_top_k=2, mode='min')
trainer = Trainer(gpus=1, callbacks=[es,checkpoint], logger=logger, max_epochs =200, gradient_clip=50)
trainer.fit(model, train_loader, val_loader)