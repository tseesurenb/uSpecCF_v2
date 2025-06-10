'''
Created on June 7, 2025
PyTorch Implementation of uSpec: Universal Spectral Collaborative Filtering

@author: Tseesuren Batsuuri (tseesuren.batsuuri@hdr.mq.edu.au)
'''

import world
import utils
import procedure
import time
from register import dataset, MODELS
import warnings
warnings.filterwarnings("ignore", message="Can't initialize NVML")

# Set random seed for reproducibility
utils.set_seed(world.seed)

print(f"\nCreating Universal Spectral CF model (seed: {world.seed}, device: {world.device})...")
model_start = time.time()
u_n_eigen = world.config.get('u_n_eigen', 0)
i_n_eigen = world.config.get('i_n_eigen', 0)
adj_mat = dataset.UserItemNet.tolil()
UniversalSpectralCF = MODELS['uspec']
Recmodel = UniversalSpectralCF(adj_mat, world.config)
print(f"Model created in {time.time() - model_start:.2f}s")

# Display model parameter information
param_info = Recmodel.get_parameter_count()
print(f"\nModel Parameters:")
print(f"  └─ Total Parameters: {param_info['total']:,}")
print(f"  └─ Filter Parameters: {param_info['filter']:,}")
print(f"  └─ Combination Parameters: {param_info['combination']:,}")
print(f"  └─ Other Parameters: {param_info['other']:,}")

# Training
print(f"\nStarting training...")
training_start = time.time()
trained_model, final_results = procedure.train_and_evaluate(dataset, Recmodel, world.config)
total_time = time.time() - training_start

# Final results summary
print(f"\n" + "="*60)
print(f"FINAL RESULTS SUMMARY")
print(f"="*60)
print(f"Model: Universal Spectral CF")
print(f"Dataset: {world.config['dataset'].upper()}")
print(f"Filter Design: {world.config.get('filter', 'enhanced_basis').upper()}")

print(f"Eigenvalue Configuration:")
if u_n_eigen > 0 and i_n_eigen > 0:
    print(f"  User Eigenvalues: {u_n_eigen}")
    print(f"  Item Eigenvalues: {i_n_eigen}")
    print(f"  Ratio (i/u): {i_n_eigen/u_n_eigen:.2f}")
else:
    if hasattr(Recmodel, 'u_n_eigen') and hasattr(Recmodel, 'i_n_eigen'):
        print(f"  Auto-adaptive eigenvalues")
        print(f"  Actual User Eigenvalues: {Recmodel.u_n_eigen}")
        print(f"  Actual Item Eigenvalues: {Recmodel.i_n_eigen}")
        print(f"  Actual Ratio (i/u): {Recmodel.i_n_eigen/Recmodel.u_n_eigen:.2f}")
    else:
        print(f"  Default eigenvalue configuration")

print(f"Total Training Time: {total_time:.2f}s")
print(f"Final Test Results: Recall@20={final_results['recall'][0]:.6f}, "
      f"Precision@20={final_results['precision'][0]:.6f}, "
      f"NDCG@20={final_results['ndcg'][0]:.6f}")
print(f"="*60)