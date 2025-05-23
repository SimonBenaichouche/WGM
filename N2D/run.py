# -*- coding: utf-8 -*-


#from conv_surrogate import *
from conv_simple_cnn import *


try:
    dataset_vorticity =torch.load("../../run_vort_1").float().view(-1,1,128,128)[:1000]
    pooled_vorticity = torch.load("../../pooled_vorticity")
    #pooled_vorticity = F.interpolate(pooled_vorticity, size=(128, 128), mode='bilinear', align_corners=False)
    future_states = torch.load("../../future_states")
except:
    pooled_vorticity = torch.load("data/pooled_tensor")
    #pooled_vorticity = F.interpolate(pooled_vorticity, size=(128, 128), mode='bilinear', align_corners=False)
    future_states = torch.load("data/future_states")

from torch.utils.data import DataLoader, Dataset
from torch.utils.data import TensorDataset, DataLoader, random_split

dataset = TensorDataset(pooled_vorticity, future_states)

# split dataset into train val and test

train_size = int(0.9 * len(dataset))
val_size = int(0.05 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Dimensions
width, height = 128, 128

# Création du masque initial
mask = torch.zeros(height, width)

# create a mask to generate a partial observation setting 
for block_col_start in range(0, width, 4):
    random_col = block_col_start + np.random.randint(0, 4)
    mask[:, random_col] = 1



'''
set mode to gradient if you want to parameterize the flow expliciterly as the gradient of a scalar function, (mandatory if you use the conv_surrogate model)
'''
#mode = "gradient"  
mode = ""
model = GradientNet(ts= 1, mask=mask,time_steps = 40, lr = 1e-5, mode = mode,step_size = 2.5e-1)

import torch
import torch.utils.data as data
import pytorch_lightning as pl

try:
    #model.load_state_dict(torch.load("surrogate_step=1"))
    model.load_state_dict(torch.load("weights_last_iter_unettanh"))
    print("model weight loaded")
except :
    print("error")
    

train_data_loader = data.DataLoader(train_dataset, batch_size=10, shuffle=True, drop_last=True, num_workers=0)
val_data_loader = data.DataLoader(val_dataset, batch_size=40, shuffle=False, drop_last=False, num_workers=0)
# configure the trainer
trainer = pl.Trainer(
    max_epochs=10,  # numbers of epoch
    accelerator='gpu',
    devices=1,  
    check_val_every_n_epoch=1, 
    accumulate_grad_batches=1  
)

# Entraîner le modèle et lancer la validation
print("Entraînement du modèle")
trainer.fit(model, train_dataloaders=train_data_loader, val_dataloaders=val_data_loader)
