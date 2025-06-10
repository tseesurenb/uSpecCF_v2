'''
Created on June 7, 2025
PyTorch Implementation of uSpec: Universal Spectral Collaborative Filtering

@author: Tseesuren Batsuuri (tseesuren.batsuuri@hdr.mq.edu.au)
'''

import torch
from parse import parse_args
import multiprocessing

args = parse_args()

config = {}
all_dataset = ['lastfm', 'gowalla', 'yelp2018', 'amazon-book', 'ml-100k']
all_models = ['uspec']

# Basic training parameters
config['train_u_batch_size'] = args.train_u_batch_size
config['eval_u_batch_size'] = args.eval_u_batch_size
config['dataset'] = args.dataset
config['lr'] = args.lr
config['decay'] = args.decay
config['epochs'] = args.epochs
config['in_mat'] = args.in_mat
config['verbose'] = args.verbose
config['val_ratio'] = args.val_ratio
config['patience'] = args.patience
config['min_delta'] = args.min_delta
config['n_epoch_eval'] = args.n_epoch_eval
config['u_n_eigen'] = args.u_n_eigen  # User eigenvalue count
config['i_n_eigen'] = args.i_n_eigen  # Item eigenvalue count
config['filter'] = args.filter
config['init_filter'] = args.init_filter
config['filter_order'] = args.filter_order
config['similarity_threshold'] = args.similarity_threshold

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
config['device'] = device

CORES = multiprocessing.cpu_count() // 2
seed = args.seed

dataset = args.dataset
model_name = args.model
if dataset not in all_dataset:
    raise NotImplementedError(f"Haven't supported {dataset} yet!, try {all_dataset}")
if model_name not in all_models:
    raise NotImplementedError(f"Haven't supported {model_name} yet!, try {all_models}")

TRAIN_epochs = args.epochs
topks = eval(args.topks)

def cprint(words: str):
    print(f"\033[0;30;43m{words}\033[0m")