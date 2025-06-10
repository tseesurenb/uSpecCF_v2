'''
Created on June 7, 2025
PyTorch Implementation of uSpec: Universal Spectral Collaborative Filtering

@author: Tseesuren Batsuuri (tseesuren.batsuuri@hdr.mq.edu.au)
'''

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Universal Spectral CF")

    # Basic training parameters
    parser.add_argument('--lr', type=float, default=0.1, help="the learning rate")
    parser.add_argument('--decay', type=float, default=1e-4, help="the weight decay for l2 normalizaton")
    parser.add_argument('--train_u_batch_size', type=int, default=1000, help='batch size for training users, -1 for full dataset')
    parser.add_argument('--eval_u_batch_size', type=int, default=500, help="batch size for evaluation users (memory management)")
    parser.add_argument('--epochs', type=int, default=50)
    
    # Dataset and evaluation
    parser.add_argument('--dataset', type=str, default='gowalla', help="available datasets: [lastfm, gowalla, yelp2018, amazon-book, ml-100k]")
    parser.add_argument('--topks', nargs='?', default="[20]", help="@k test list")
    parser.add_argument('--val_ratio', type=float, default=0.1, help='ratio of training data to use for validation (0.1 = 10%)')
    
    # Model architecture
    parser.add_argument('--model', type=str, default='uspec', help='rec-model, support [uspec]')
    parser.add_argument('--in_mat', type=str, default='ui', help='u, i, or ui')
    parser.add_argument('--similarity_threshold', type=float, default=0.0, help='Threshold for filtering weak similarities (maintains symmetry). The default is 0.0, which means no filtering.')
    parser.add_argument('--u_n_eigen', type=int, default=0, help='Number of eigenvalues for user similarity matrix (0 = auto-adaptive)')
    parser.add_argument('--i_n_eigen', type=int, default=0, help='Number of eigenvalues for item similarity matrix (0 = auto-adaptive)')
    
    
    # Filter design options
    parser.add_argument('--filter', type=str, default='enhanced_basis', 
                       choices=['original', 'basis', 'enhanced_basis', 'adaptive_golden', 
                               'multiscale', 'ensemble', 'band_stop', 'adaptive_band_stop', 
                               'parametric_multi_band', 'harmonic'], help='Filter design')
    parser.add_argument('--init_filter', type=str, default='smooth', help='Initial filter pattern')
    parser.add_argument('--filter_order', type=int, default=6, help='polynomial order for spectral filters')

    # Training control
    parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
    parser.add_argument('--min_delta', type=float, default=1e-5, help='minimum improvement for early stopping')
    parser.add_argument('--n_epoch_eval', type=int, default=5, help='evaluate every N epochs')
    
    # Experiment control
    parser.add_argument('--seed', type=int, default=2025, help='random seed')
    parser.add_argument('--verbose', type=int, default=1, help='0 for silent, 1 for verbose')
    
    return parser.parse_args()