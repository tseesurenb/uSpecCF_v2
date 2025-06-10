'''
Enhanced Model Registration - Supports Multi-View Universal Spectral CF
'''

import world
import dataloader

# Dataset loading (unchanged)
if world.dataset in ['gowalla', 'yelp2018', 'amazon-book']:
    dataset = dataloader.Loader(path="../data/"+world.dataset)
elif world.dataset == 'lastfm':
    dataset = dataloader.LastFM()
elif world.dataset == 'ml-100k':
    dataset = dataloader.ML100K()
else:
    raise ValueError(f"Unknown dataset: {world.dataset}")

# Import both original and enhanced models
try:
    import model  # Original uSpec
    import m_model  # Enhanced Multi-View uSpec
    print(f"Available model classes:")
    print(f"  Original: {[attr for attr in dir(model) if 'SpectralCF' in attr]}")
    print(f"  Enhanced: {[attr for attr in dir(m_model) if 'SpectralCF' in attr]}")
except Exception as e:
    print(f"Error importing model modules: {e}")
    raise

# Enhanced model registration
MODELS = {}

# Register original uSpec models
for class_name in ['UniversalSpectralCF', 'SimplifiedSpectralCF']:
    if hasattr(model, class_name):
        ModelClass = getattr(model, class_name)
        MODELS['uspec'] = ModelClass
        MODELS['original_uspec'] = ModelClass
        print(f"Registered original uSpec: {class_name}")
        break

# Register enhanced multi-view models
for class_name in ['MultiViewUniversalSpectralCF', 'EnhancedUniversalSpectralCF']:
    if hasattr(m_model, class_name):
        ModelClass = getattr(m_model, class_name)
        MODELS['multiview_uspec'] = ModelClass
        MODELS['enhanced_uspec'] = ModelClass
        MODELS['mv_uspec'] = ModelClass
        print(f"Registered multi-view uSpec: {class_name}")
        break

# Auto-select model based on configuration
model_selection = world.config.get('model', 'multiview_uspec')

# Check if multi-view features are requested
use_multiview = (
    world.config.get('use_adjacency_views', True) or 
    world.config.get('use_similarity_views', True) or
    len(world.config.get('gamma_values', [])) > 1
)

if use_multiview and 'multiview_uspec' in MODELS:
    # Use enhanced model for multi-view configurations
    if model_selection in ['uspec', 'original_uspec']:
        print(f"🎭 Multi-view features detected, switching to enhanced model")
        world.config['model'] = 'multiview_uspec'
elif not use_multiview and model_selection.startswith('multiview'):
    # Use original model for single-view configurations
    print(f"📊 Single-view configuration, using original model")
    world.config['model'] = 'uspec'

# Ensure we have the requested model
final_model = world.config['model']
if final_model not in MODELS:
    available_models = list(MODELS.keys())
    print(f"Warning: Model '{final_model}' not found")
    print(f"Available models: {available_models}")
    if available_models:
        fallback_model = available_models[0]
        print(f"Using fallback model: {fallback_model}")
        world.config['model'] = fallback_model
        final_model = fallback_model
    else:
        raise RuntimeError("No models available")

# Display final configuration
print(f"\n📊 Final Model Configuration:")
print(f"  Selected model: {final_model} -> {MODELS[final_model].__name__}")
print(f"  Dataset: {world.dataset}")
print(f"  Users: {dataset.n_users:,}, Items: {dataset.m_items:,}")
print(f"  Training: {dataset.trainDataSize:,}, Test: {len(dataset.testDict):,}")

# Multi-view specific configuration display
if final_model.startswith('multiview') or final_model.startswith('enhanced'):
    print(f"\n🎭 Multi-View Configuration:")
    print(f"  Adjacency views: {world.config.get('use_adjacency_views', True)}")
    print(f"  Similarity views: {world.config.get('use_similarity_views', True)}")
    print(f"  Gamma values: {world.config.get('gamma_values', [0.0, 0.5, 1.0])}")
    print(f"  Similarity threshold: {world.config.get('similarity_threshold', 0.0)}")
    
    # Calculate total views
    total_views = 0
    if world.config.get('use_adjacency_views', True):
        total_views += len(world.config.get('gamma_values', [0.0, 0.5, 1.0])) * 2
    if world.config.get('use_similarity_views', True):
        total_views += 2
    print(f"  Total views: {total_views}")

print(f"  Filter design: {world.config.get('filter', 'enhanced_basis')}")
print(f"  Filter order: {world.config.get('filter_order', 6)}")
print(f"  Device: {world.device}")
print("="*50)