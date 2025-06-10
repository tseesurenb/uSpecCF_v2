'''
Enhanced World Configuration for Multi-View Universal Spectral CF
'''

import torch
from parse import parse_args
import multiprocessing

args = parse_args()

config = {}
all_dataset = ['lastfm', 'gowalla', 'yelp2018', 'amazon-book', 'ml-100k']
all_models = ['uspec', 'multiview_uspec', 'enhanced_uspec', 'mv_uspec']

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

# Multi-view configuration (NEW)
config['use_adjacency_views'] = args.use_adjacency_views
config['use_similarity_views'] = args.use_similarity_views
config['gamma_values'] = args.gamma_values
config['view_combination'] = args.view_combination
config['view_weight_regularization'] = args.view_weight_regularization

# Spectral filtering parameters
config['similarity_threshold'] = args.similarity_threshold
config['u_n_eigen'] = args.u_n_eigen
config['i_n_eigen'] = args.i_n_eigen
config['filter'] = args.filter
config['init_filter'] = args.init_filter
config['filter_order'] = args.filter_order

# Model selection
config['model'] = args.model

# Experimental parameters
config['ablation_study'] = args.ablation_study

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
config['device'] = device

CORES = multiprocessing.cpu_count() // 2
seed = args.seed

dataset = args.dataset
model_name = args.model

# Validation
if dataset not in all_dataset:
    raise NotImplementedError(f"Haven't supported {dataset} yet!, try {all_dataset}")

if model_name not in all_models:
    print(f"Warning: Model {model_name} not in standard list, will try to load anyway")
    print(f"Standard models: {all_models}")

TRAIN_epochs = args.epochs
topks = eval(args.topks)

def cprint(words: str):
    print(f"\033[0;30;43m{words}\033[0m")

# Auto-configuration based on dataset and views
def auto_configure_views():
    """Auto-configure view settings based on dataset characteristics"""
    n_users = 1000  # Placeholder, will be updated from dataset
    n_items = 1000  # Placeholder, will be updated from dataset
    
    # Adjust gamma values based on dataset
    if config['dataset'] == 'ml-100k':
        # Small dataset, can handle more views
        if len(config['gamma_values']) == 3 and config['gamma_values'] == [0.0, 0.5, 1.0]:
            config['gamma_values'] = [0.0, 0.3, 0.5, 0.7, 1.0]  # More fine-grained
    elif config['dataset'] in ['gowalla', 'yelp2018']:
        # Medium datasets, keep standard gamma values
        pass
    elif config['dataset'] == 'amazon-book':
        # Large dataset, might need fewer views for efficiency
        if len(config['gamma_values']) > 3:
            config['gamma_values'] = [0.0, 0.5, 1.0]  # Keep it simple
    
    # Ensure at least one view type is enabled
    if not config['use_adjacency_views'] and not config['use_similarity_views']:
        print("Warning: No views enabled, enabling similarity views as default")
        config['use_similarity_views'] = True

# Apply auto-configuration
auto_configure_views()

# Print comprehensive configuration
print(f"\n=== Multi-View Universal Spectral CF Configuration ===")
print(f"Dataset: {config['dataset']}")
print(f"Model: {config['model']}")
print(f"Device: {device}")

print(f"\n🎭 View Configuration:")
print(f"  Adjacency views (PolyCF): {config['use_adjacency_views']}")
if config['use_adjacency_views']:
    print(f"    Gamma values: {config['gamma_values']}")
    print(f"    Total adjacency views: {len(config['gamma_values']) * 2}")

print(f"  Similarity views (uSpec): {config['use_similarity_views']}")
if config['use_similarity_views']:
    print(f"    Similarity threshold: {config['similarity_threshold']}")
    print(f"    Total similarity views: 2")

# Calculate total views
total_views = 0
if config['use_adjacency_views']:
    total_views += len(config['gamma_values']) * 2  # User and item gram for each gamma
if config['use_similarity_views']:
    total_views += 2  # User and item similarity
print(f"  Total views: {total_views}")

print(f"\n🔧 Spectral Filtering:")
print(f"  Filter design: {config['filter']}")
print(f"  Filter initialization: {config['init_filter']}")
print(f"  Filter order: {config['filter_order']}")
print(f"  Eigenvalues: u={config['u_n_eigen']}, i={config['i_n_eigen']} (0=auto)")

print(f"\n⚙️ Training:")
print(f"  Epochs: {config['epochs']}")
print(f"  Learning rate: {config['lr']}")
print(f"  Weight decay: {config['decay']}")
print(f"  View combination: {config['view_combination']}")
if config['view_weight_regularization'] > 0:
    print(f"  View weight regularization: {config['view_weight_regularization']}")

if config['ablation_study']:
    print(f"\n🧪 Ablation Study: {config['ablation_study']}")

print("=" * 55)