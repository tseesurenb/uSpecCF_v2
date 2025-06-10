'''
Enhanced Main Script for Multi-View Universal Spectral CF

@author: Tseesuren Batsuuri (tseesuren.batsuuri@hdr.mq.edu.au)
'''

import world
import utils
import procedure
import time
from register import dataset, MODELS
import torch

import warnings
warnings.filterwarnings("ignore", message="Can't initialize NVML")

# Set random seed for reproducibility
utils.set_seed(world.seed)

print(f"\nCreating Multi-View Universal Spectral CF model (seed: {world.seed}, device: {world.device})...")
model_start = time.time()

# Get model class and create model
ModelClass = MODELS[world.config['model']]
adj_mat = dataset.UserItemNet.tolil()
Recmodel = ModelClass(adj_mat, world.config)
print(f"Model created in {time.time() - model_start:.2f}s")

# Display model parameter information
if hasattr(Recmodel, 'get_parameter_count'):
    param_info = Recmodel.get_parameter_count()
    print(f"\nModel Parameters:")
    print(f"  └─ Total Parameters: {param_info['total']:,}")
    
    if 'view_filters' in param_info:
        # Multi-view model
        print(f"  └─ View Filter Parameters: {param_info['view_filters']:,}")
        print(f"  └─ View Combination Parameters: {param_info['view_combination']:,}")
        print(f"  └─ Other Parameters: {param_info.get('other', 0):,}")
    else:
        # Original model
        print(f"  └─ Filter Parameters: {param_info.get('filter', 0):,}")
        print(f"  └─ Combination Parameters: {param_info.get('combination', 0):,}")
        print(f"  └─ Other Parameters: {param_info.get('other', 0):,}")

# Display view configuration for multi-view models
if hasattr(Recmodel, 'view_filters'):
    print(f"\n🎭 Multi-View Configuration:")
    print(f"  └─ Total Views: {len(Recmodel.view_filters)}")
    
    # Display view details
    adjacency_views = [name for name in Recmodel.view_filters.keys() if 'gamma' in name]
    similarity_views = [name for name in Recmodel.view_filters.keys() if 'similarity' in name]
    
    if adjacency_views:
        print(f"  └─ Adjacency Views ({len(adjacency_views)}):")
        for view_name in adjacency_views:
            gamma_val = view_name.split('gamma_')[1] if 'gamma_' in view_name else 'unknown'
            view_type = 'User Gram' if 'user' in view_name else 'Item Gram'
            print(f"     ├─ {view_type} (γ={gamma_val})")
    
    if similarity_views:
        print(f"  └─ Similarity Views ({len(similarity_views)}):")
        for view_name in similarity_views:
            view_type = view_name.replace('_', ' ').title()
            print(f"     ├─ {view_type}")
    
    # Display eigenvalue configuration
    if hasattr(Recmodel, 'u_n_eigen') and hasattr(Recmodel, 'i_n_eigen'):
        print(f"\n🧮 Eigenvalue Configuration:")
        print(f"  └─ User Eigenvalues: {Recmodel.u_n_eigen}")
        print(f"  └─ Item Eigenvalues: {Recmodel.i_n_eigen}")
        print(f"  └─ Ratio (i/u): {Recmodel.i_n_eigen/Recmodel.u_n_eigen:.2f}")

# Display filter information
filter_info = world.config.get('filter', 'enhanced_basis')
init_filter = world.config.get('init_filter', 'smooth')
print(f"\n🔧 Filter Configuration:")
print(f"  └─ Filter Design: {filter_info.upper()}")
print(f"  └─ Filter Initialization: {init_filter.upper()}")
print(f"  └─ Filter Order: {world.config.get('filter_order', 6)}")

# Training
print(f"\nStarting training...")
training_start = time.time()
trained_model, final_results = procedure.train_and_evaluate(dataset, Recmodel, world.config)
total_time = time.time() - training_start

# Final results summary
print(f"\n" + "="*60)
print(f"FINAL RESULTS SUMMARY")
print(f"="*60)
print(f"Model: {ModelClass.__name__}")
print(f"Dataset: {world.config['dataset'].upper()}")

# Multi-view specific summary
if hasattr(trained_model, 'view_filters'):
    print(f"Configuration: Multi-View Spectral CF")
    print(f"  └─ Total Views: {len(trained_model.view_filters)}")
    print(f"  └─ Adjacency Views: {world.config.get('use_adjacency_views', False)}")
    print(f"  └─ Similarity Views: {world.config.get('use_similarity_views', False)}")
    if world.config.get('use_adjacency_views', False):
        print(f"  └─ Gamma Values: {world.config.get('gamma_values', [])}")
    if world.config.get('similarity_threshold', 0) > 0:
        print(f"  └─ Similarity Threshold: {world.config.get('similarity_threshold', 0)}")
else:
    print(f"Filter Design: {world.config.get('filter', 'enhanced_basis').upper()}")

# Eigenvalue summary
u_n_eigen = world.config.get('u_n_eigen', 0)
i_n_eigen = world.config.get('i_n_eigen', 0)
if u_n_eigen > 0 and i_n_eigen > 0:
    print(f"Eigenvalue Configuration:")
    print(f"  └─ User Eigenvalues: {u_n_eigen}")
    print(f"  └─ Item Eigenvalues: {i_n_eigen}")
    print(f"  └─ Ratio (i/u): {i_n_eigen/u_n_eigen:.2f}")
else:
    if hasattr(trained_model, 'u_n_eigen') and hasattr(trained_model, 'i_n_eigen'):
        print(f"Eigenvalue Configuration:")
        print(f"  └─ Auto-adaptive eigenvalues")
        print(f"  └─ Actual User Eigenvalues: {trained_model.u_n_eigen}")
        print(f"  └─ Actual Item Eigenvalues: {trained_model.i_n_eigen}")
        print(f"  └─ Actual Ratio (i/u): {trained_model.i_n_eigen/trained_model.u_n_eigen:.2f}")

print(f"Total Training Time: {total_time:.2f}s")
print(f"Final Test Results: Recall@20={final_results['recall'][0]:.6f}, "
      f"Precision@20={final_results['precision'][0]:.6f}, "
      f"NDCG@20={final_results['ndcg'][0]:.6f}")

# Display learned view weights if available
if hasattr(trained_model, 'view_combination_weights'):
    weights = torch.softmax(trained_model.view_combination_weights, dim=0)
    print(f"\n🎭 Learned View Combination Weights:")
    if hasattr(trained_model, 'view_filters'):
        for i, (view_name, weight) in enumerate(zip(trained_model.view_filters.keys(), weights.cpu())):
            print(f"  └─ {view_name}: {weight:.4f}")
    else:
        print(f"  └─ Weights: {weights.detach().cpu().numpy()}")

print(f"="*60)