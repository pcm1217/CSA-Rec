import os
import sys
import argparse

from utils import *
from train_model import *

#os.environ['CUDA_MPS_PIPE_DIRECTORY'] = '/nonexistent'


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # GPU train options
    parser.add_argument("--multi_gpu", action='store_true')
    parser.add_argument('--gpu_num', default=4,type=int)
    
    # model setting
    parser.add_argument("--llm", type=str, default='llama', help='opt')
    parser.add_argument("--recsys", type=str, default='sasrec')
    
    # dataset setting
    parser.add_argument("--rec_pre_trained_data", type=str, default='TN')
    
    # train phase setting
    parser.add_argument("--pretrain_stage1", action='store_true')
    parser.add_argument("--pretrain_stage2", action='store_false')
    parser.add_argument("--inference", action='store_false')
    
    # hyperparameters options
    parser.add_argument('--batch_size1', default=16, type=int)
    parser.add_argument('--batch_size2', default=2, type=int)
    parser.add_argument('--batch_size_infer', default=2, type=int)
    parser.add_argument('--maxlen', default=20, type=int)
    parser.add_argument('--num_epochs', default=10, type=int)
    parser.add_argument("--stage1_lr", type=float, default=0.0001)
    parser.add_argument("--stage2_lr", type=float, default=0.0001)
    
    args = parser.parse_args()
    
    args.device = 'cuda:' + str(args.gpu_num)
    
    if args.pretrain_stage1:
        train_model_phase1(args)
    if args.pretrain_stage2:
        train_model_phase2(args)
    if args.inference:
        inference(args)