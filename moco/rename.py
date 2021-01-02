from collections import OrderedDict
import torch
import os

checkpoint = torch.load("<MODEL_PATH_(A_TAR_FILE)>")
model = checkpoint['state_dict']

encoder_dict = OrderedDict()
tcn_dict = OrderedDict()

for key in model.keys():
    if 'encoder_q' in key:
        
        if 'c3d' in key:
            new_key = 'module.' + key[21:]
            encoder_dict[new_key] = model[key]
        
        elif 'tcn' in key:
            new_key = key[21:]
            tcn_dict[new_key] = model[key]

torch.save(encoder_dict, '<PATH_TO_SAVE_C3D>')
torch.save(tcn_dict, '<PATH_TO_SAVE_TCN')
