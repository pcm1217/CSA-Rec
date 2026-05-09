import contextlib
import logging
import os
import glob

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F

from utils import *
from pre_train.model import STRec


def load_checkpoint(recsys, pre_trained):
    path = f''
    
    pth_file_path = find_filepath(path, '.pth')
    assert len(pth_file_path) == 1, 'There are more than two models in this dir. You need to remove other model files.\n'
    ckpt_path = pth_file_path[0]
    
    try:
        kwargs, checkpoint = torch.load(
            ckpt_path,
            map_location="cpu",
            weights_only=False
        )
    except TypeError:
        kwargs, checkpoint = torch.load(ckpt_path, map_location="cpu")
    logging.info("load checkpoint from %s" % pth_file_path[0])

    return kwargs, checkpoint

class RecSys(nn.Module):
    def __init__(self, recsys_model, pre_trained_data, device):
        super().__init__()
        kwargs, checkpoint = load_checkpoint(recsys_model, pre_trained_data)
        kwargs['args'].device = device
        model = STRec(**kwargs)
        model.load_state_dict(checkpoint)
            
        for p in model.parameters():
            p.requires_grad = False
            
        self.poi_num = model.poi_num
        self.user_num = model.user_num
        self.model = model.to(device)
        self.hidden_units = kwargs['args'].hidden_units
        
    def forward():
        print('forward')