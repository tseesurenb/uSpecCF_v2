'''
Enhanced Argument Parser for Multi-View Universal Spectral CF
'''

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Multi-View Universal Spectral CF")

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
    parser.add_argument('--model', type=str, default='multiview_uspec', 
                       help='model type: [uspec, multiview_uspec, enhanced_uspec, mv_uspec]')
    parser.add_argument('--in_mat', type=str, default='ui', help='u, i, or ui (legacy parameter)')
    
    # Multi-view configuration (NEW)
    parser.add_argument('--use_adjacency_views', action='store_true', default=True,
                       help='enable PolyCF-style adjacency views with different gamma normalizations')
    parser.add_argument('--use_similarity_views', action='store_true', default=True,
                       help='enable uSpec-style similarity views (user-user, item-item)')
    parser.add_argument('--no_adjacency_views', dest='use_adjacency_views', action='store_false',
                       help='disable adjacency views')
    parser.add_argument('--no_similarity_views', dest='use_similarity_views', action='store_false',
                       help='disable similarity views')
    
    # PolyCF-style gamma values for adjacency normalization
    parser.add_argument('--gamma_values', type=float, nargs='+', default=[0.0, 0.5, 1.0],
                       help='gamma values for PolyCF adjacency normalization (default: [0.0, 0.5, 1.0])')
    parser.add_argument('--single_gamma', type=float, default=None,
                       help='use only single gamma value (for ablation studies)')
    
    # Similarity processing (uSpec-style)
    parser.add_argument('--similarity_threshold', type=float, default=0.0, 
                       help='threshold for filtering weak similarities (maintains symmetry)')
    parser.add_argument('--u_n_eigen', type=int, default=0, 
                       help='number of eigenvalues for user matrices (0 = auto-adaptive)')
    parser.add_argument('--i_n_eigen', type=int, default=0, 
                       help='number of eigenvalues for item matrices (0 = auto-adaptive)')
    
    # Filter design options
    parser.add_argument('--filter', type=str, default='enhanced_basis', 
                       choices=['original', 'basis', 'enhanced_basis', 'adaptive_golden', 
                               'multiscale', 'ensemble', 'band_stop', 'adaptive_band_stop', 
                               'parametric_multi_band', 'harmonic'], 
                       help='spectral filter design')
    parser.add_argument('--init_filter', type=str, default='smooth', 
                       help='initial filter pattern')
    parser.add_argument('--filter_order', type=int, default=6, 
                       help='polynomial order for spectral filters')

    # Training control
    parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
    parser.add_argument('--min_delta', type=float, default=1e-5, help='minimum improvement for early stopping')
    parser.add_argument('--n_epoch_eval', type=int, default=5, help='evaluate every N epochs')
    
    # Experiment control
    parser.add_argument('--seed', type=int, default=2025, help='random seed')
    parser.add_argument('--verbose', type=int, default=1, help='0 for silent, 1 for verbose')
    
    # View combination strategies (ADVANCED)
    parser.add_argument('--view_combination', type=str, default='learned_weights', 
                       choices=['learned_weights', 'equal_weights', 'adjacency_only', 'similarity_only'],
                       help='strategy for combining multiple views')
    parser.add_argument('--view_weight_regularization', type=float, default=0.0,
                       help='L2 regularization on view combination weights')
    
    # Experimental configurations
    parser.add_argument('--ablation_study', type=str, default=None,
                       choices=['gamma_range', 'view_types', 'filter_designs', 'threshold_values'],
                       help='run specific ablation study')
    
    # Backward compatibility
    parser.add_argument('--filter_design', type=str, default=None,
                       help='DEPRECATED: use --filter instead')
    
    args = parser.parse_args()
    
    # Handle backward compatibility
    if args.filter_design is not None:
        print(f"Warning: --filter_design is deprecated, use --filter instead")
        args.filter = args.filter_design
    
    # Handle single gamma override
    if args.single_gamma is not None:
        args.gamma_values = [args.single_gamma]
        print(f"Using single gamma value: {args.single_gamma}")
    
    # Auto-adjust model selection based on view configuration
    if not args.use_adjacency_views and not args.use_similarity_views:
        print("Warning: Both adjacency and similarity views disabled, enabling similarity views")
        args.use_similarity_views = True
    
    return args