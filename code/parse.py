'''
Created on June 7, 2025
PyTorch Implementation of uSpec: Universal Spectral Collaborative Filtering
Simplified argument parser for enhanced model only

@author: Tseesuren Batsuuri (tseesuren.batsuuri@hdr.mq.edu.au)
'''

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Enhanced Universal Spectral CF with Matrix Multiplication Similarity")

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
    parser.add_argument('--u_n_eigen', type=int, default=0, 
                       help='Number of eigenvalues for user similarity matrix (0 = auto-adaptive)')
    parser.add_argument('--i_n_eigen', type=int, default=0, 
                       help='Number of eigenvalues for item similarity matrix (0 = auto-adaptive)')
    parser.add_argument('--model', type=str, default='uspec', help='rec-model, support [uspec]')
    parser.add_argument('--filter', type=str, default='ui', help='u, i, or ui')
    parser.add_argument('--filter_order', type=int, default=6, help='polynomial order for spectral filters')
    
    # Filter design options
    parser.add_argument('--filter_design', type=str, default='enhanced_basis', 
                       choices=['original', 'basis', 'enhanced_basis', 'adaptive_golden', 
                               'multiscale', 'ensemble', 'band_stop', 'adaptive_band_stop', 
                               'parametric_multi_band', 'harmonic'], 
                       help='Filter design')
    parser.add_argument('--init_filter', type=str, default='smooth', 
                       help='Initial filter pattern')
    
    # Similarity parameters (matrix multiplication only)
    parser.add_argument('--similarity_threshold', type=float, default=0.01,
                       help='Threshold for filtering weak similarities (maintains symmetry)')
    
    # Training control
    parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
    parser.add_argument('--min_delta', type=float, default=1e-5, help='minimum improvement for early stopping')
    parser.add_argument('--n_epoch_eval', type=int, default=5, help='evaluate every N epochs')
    
    # Experiment control
    parser.add_argument('--seed', type=int, default=2025, help='random seed')
    parser.add_argument('--verbose', type=int, default=1, help='0 for silent, 1 for verbose')
    
    return parser.parse_args()


def validate_args(args):
    """Enhanced argument validation for enhanced model only"""
    
    print(f"🚀 Using Enhanced Universal Spectral CF with Matrix Multiplication Similarity")
    print(f"   Filter Design: {args.filter_design}")
    print(f"   Init Filter: {args.init_filter}")
    
    # Eigenvalue parameter validation
    if args.u_n_eigen > 0 and args.i_n_eigen > 0:
        print(f"✅ Using separate eigenvalue counts: u_n_eigen={args.u_n_eigen}, i_n_eigen={args.i_n_eigen}")
    else:
        print(f"🤖 Using auto-adaptive eigenvalue counts (recommended)")
    
    # Dataset-specific recommendations
    dataset_recommendations = {
        'ml-100k': {'u_n_eigen': 48, 'i_n_eigen': 64},
        'ml-1m': {'u_n_eigen': 96, 'i_n_eigen': 128},
        'lastfm': {'u_n_eigen': 64, 'i_n_eigen': 96},
        'gowalla': {'u_n_eigen': 128, 'i_n_eigen': 256},
        'yelp2018': {'u_n_eigen': 192, 'i_n_eigen': 384},
        'amazon-book': {'u_n_eigen': 256, 'i_n_eigen': 512}
    }
    
    if args.dataset in dataset_recommendations:
        rec = dataset_recommendations[args.dataset]
        if args.u_n_eigen == 0 and args.i_n_eigen == 0:
            print(f"💡 {args.dataset} recommendation: --u_n_eigen {rec['u_n_eigen']} --i_n_eigen {rec['i_n_eigen']}")
    
    return args


if __name__ == "__main__":
    # Demo the enhanced model functionality
    print("=== ENHANCED MODEL EXAMPLES ===")
    print()
    print("# Auto-adaptive eigenvalues:")
    print("python main.py --dataset ml-100k --filter_design enhanced_basis")
    print()
    print("# Manual eigenvalues:")
    print("python main.py --dataset ml-100k --u_n_eigen 48 --i_n_eigen 64 --filter_design multiscale")
    print()
    print("# Large dataset with ensemble filter:")
    print("python main.py --dataset gowalla --u_n_eigen 128 --i_n_eigen 256 --filter_design ensemble")
    print()
    print("# Custom similarity threshold:")
    print("python main.py --dataset lastfm --similarity_threshold 0.005 --filter_design harmonic")