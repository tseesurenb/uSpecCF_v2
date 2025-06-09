'''
Created on June 7, 2025
PyTorch Implementation of uSpec: Universal Spectral Collaborative Filtering
Simplified register for enhanced model only

@author: Tseesuren Batsuuri (tseesuren.batsuuri@hdr.mq.edu.au)
'''
import world
import dataloader

# Dataset loading
if world.dataset in ['gowalla', 'yelp2018', 'amazon-book']:
    dataset = dataloader.Loader(path="../data/"+world.dataset)
elif world.dataset == 'lastfm':
    dataset = dataloader.LastFM()
elif world.dataset == 'ml-100k':
    dataset = dataloader.ML100K()
else:
    raise ValueError(f"Unknown dataset: {world.dataset}")

# Only enhanced model supported now
import model_enhanced
MODELS = {'uspec': model_enhanced.UniversalSpectralCF}

print("🚀 Using Enhanced Universal Spectral CF with Matrix Multiplication Similarity")
print(f"   └─ Advanced filter designs and caching")
print(f"   └─ Adaptive eigenvalue calculation")
print(f"   └─ Matrix multiplication similarity (A @ A.T)")

# Display configuration info
if world.config['verbose'] > 0:
    print(f"\n📊 Dataset Configuration:")
    print(f"   └─ Dataset: {world.dataset}")
    print(f"   └─ Users: {dataset.n_users:,}, Items: {dataset.m_items:,}")
    print(f"   └─ Training: {dataset.trainDataSize:,}, Validation: {dataset.valDataSize:,}")
    
    print(f"\n⚙️  Model Configuration:")
    
    # Eigenvalue configuration
    u_n_eigen = world.config.get('u_n_eigen', 0)
    i_n_eigen = world.config.get('i_n_eigen', 0)
    
    if u_n_eigen > 0 and i_n_eigen > 0:
        print(f"   └─ User Eigenvalues: {u_n_eigen}")
        print(f"   └─ Item Eigenvalues: {i_n_eigen}")
        print(f"   └─ Eigenvalue Ratio (i/u): {i_n_eigen/u_n_eigen:.2f}")
    else:
        print(f"   └─ Eigenvalues: Auto-adaptive")
    
    print(f"   └─ Filter Design: {world.config.get('filter_design', 'enhanced_basis')}")
    print(f"   └─ Similarity Type: Matrix Multiplication")
    print(f"   └─ Similarity Threshold: {world.config.get('similarity_threshold', 0.01)}")
    print(f"   └─ Filter Type: {world.config['filter']}")
    print(f"   └─ Filter Order: {world.config['filter_order']}")
    print(f"   └─ Device: {world.device}")