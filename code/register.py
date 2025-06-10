'''
Created on June 7, 2025
PyTorch Implementation of uSpec: Universal Spectral Collaborative Filtering

@author: Tseesuren Batsuuri (tseesuren.batsuuri@hdr.mq.edu.au)
'''
import world
import dataloader
import model

# Dataset loading
if world.dataset in ['gowalla', 'yelp2018', 'amazon-book']:
    dataset = dataloader.Loader(path="../data/"+world.dataset)
elif world.dataset == 'lastfm':
    dataset = dataloader.LastFM()
elif world.dataset == 'ml-100k':
    dataset = dataloader.ML100K()
else:
    raise ValueError(f"Unknown dataset: {world.dataset}")

MODELS = {'uspec': model.UniversalSpectralCF}

# Display configuration info
if world.config['verbose'] > 0:
    print(f"\n📊 Configuration:")
    print(f"   └─ Dataset: {world.dataset}")
    print(f"   └─ Users: {dataset.n_users:,}, Items: {dataset.m_items:,}")
    print(f"   └─ Training: {dataset.trainDataSize:,}, Validation: {dataset.valDataSize:,}, Test: {len(dataset.testDict):,}")
    print(f"   └─ Filter Design: {world.config.get('filter_design', 'enhanced_basis')}")
    print(f"   └─ Similarity Threshold: {world.config.get('similarity_threshold', 0.01)}")
    print(f"   └─ Filter Type: {world.config['filter']}")
    print(f"   └─ Filter Order: {world.config['filter_order']}")
    print(f"   └─ Device: {world.device}")