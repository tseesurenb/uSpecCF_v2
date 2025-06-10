#!/usr/bin/env python3
"""
Multi-View Hyperparameter Search for ML-100K
Tests view configurations, thresholds, and eigenvalues
"""

import sys
import time
import json
from datetime import datetime

# Hack to bypass argument parsing
sys.argv = ['hp_search.py', '--dataset', 'ml-100k']

# Now import after setting argv
import world
import utils
import procedure
from register import dataset, MODELS

def run_experiment(config_dict, name="test"):
    """Run one experiment"""
    print(f"🚀 {name}")
    
    # Set config
    for key, value in config_dict.items():
        world.config[key] = value
    
    utils.set_seed(config_dict.get('seed', 2025))
    
    try:
        # Create model
        adj_mat = dataset.UserItemNet.tolil()
        ModelClass = MODELS[world.config.get('model', 'multiview_uspec')]
        model = ModelClass(adj_mat, world.config)
        
        # Train
        start = time.time()
        trained_model, results = procedure.train_and_evaluate(dataset, model, world.config)
        runtime = time.time() - start
        
        ndcg = results['ndcg'][0]
        recall = results['recall'][0] 
        precision = results['precision'][0]
        
        print(f"✅ {name}: NDCG@20: {ndcg:.6f}, Recall@20: {recall:.6f}, Precision@20: {precision:.6f} ({runtime:.1f}s)")
        return ndcg, True
        
    except Exception as e:
        print(f"❌ {name} failed: {str(e)}")
        return 0.0, False

def stage0_view_configurations():
    """Stage 0: Test different view configurations"""
    print("\n" + "="*80)
    print("🎭 STAGE 0: Multi-View Configuration Search")
    print("="*80)
    
    view_configs = [
        # Pure approaches
        {
            'name': 'adjacency_only_basic',
            'use_adjacency_views': True,
            'use_similarity_views': False,
            'gamma_values': [0.0, 0.5, 1.0],
            'filter': 'enhanced_basis'
        },
        {
            'name': 'adjacency_only_extended',
            'use_adjacency_views': True,
            'use_similarity_views': False,
            'gamma_values': [0.0, 0.3, 0.5, 0.7, 1.0],
            'filter': 'enhanced_basis'
        },
        {
            'name': 'similarity_only',
            'use_adjacency_views': False,
            'use_similarity_views': True,
            'gamma_values': [],
            'filter': 'enhanced_basis',
            'similarity_threshold': 0.01
        },
        
        # Hybrid approaches
        {
            'name': 'hybrid_basic',
            'use_adjacency_views': True,
            'use_similarity_views': True,
            'gamma_values': [0.5],
            'filter': 'enhanced_basis',
            'similarity_threshold': 0.01
        },
        {
            'name': 'hybrid_standard',
            'use_adjacency_views': True,
            'use_similarity_views': True,
            'gamma_values': [0.0, 0.5, 1.0],
            'filter': 'enhanced_basis',
            'similarity_threshold': 0.01
        },
        {
            'name': 'hybrid_extended',
            'use_adjacency_views': True,
            'use_similarity_views': True,
            'gamma_values': [0.0, 0.3, 0.7, 1.0],
            'filter': 'enhanced_basis',
            'similarity_threshold': 0.005
        },
        
        # Filter variations
        {
            'name': 'multiscale_hybrid',
            'use_adjacency_views': True,
            'use_similarity_views': True,
            'gamma_values': [0.0, 0.5, 1.0],
            'filter': 'multiscale',
            'similarity_threshold': 0.01
        },
        {
            'name': 'ensemble_hybrid',
            'use_adjacency_views': True,
            'use_similarity_views': True,
            'gamma_values': [0.0, 0.5, 1.0],
            'filter': 'ensemble',
            'similarity_threshold': 0.01
        }
    ]
    
    best_score = 0
    best_config = None
    results = []
    
    print(f"Testing {len(view_configs)} view configurations")
    print("This reveals the optimal combination of adjacency and similarity views")
    
    for i, config in enumerate(view_configs):
        print(f"\n[{i+1}/{len(view_configs)}] Testing {config['name']}...")
        
        # Create full config
        full_config = {
            'epochs': 15,  # Medium training for view comparison
            'lr': 0.1,
            'decay': 1e-4,
            'verbose': 0,
            'seed': 2025,
            'model': 'multiview_uspec'
        }
        full_config.update(config)
        
        ndcg, success = run_experiment(full_config, config['name'])
        results.append((config['name'], config, ndcg, success))
        
        if success and ndcg > best_score:
            best_score = ndcg
            best_config = config.copy()
            print(f"🎉 NEW BEST: {config['name']} - {ndcg:.6f}")
    
    # Show all results
    print(f"\n✅ Stage 0 Complete - View Configuration Results:")
    print("="*100)
    
    results.sort(key=lambda x: x[2], reverse=True)
    
    print(f"{'Rank':<4} {'Status':<6} {'Configuration':<25} {'Views':<15} {'Filter':<15} {'NDCG@20':<10}")
    print("-" * 100)
    
    for i, (name, config, score, success) in enumerate(results):
        status = "✅" if success else "❌"
        adj_views = len(config.get('gamma_values', [])) * 2 if config.get('use_adjacency_views', False) else 0
        sim_views = 2 if config.get('use_similarity_views', False) else 0
        total_views = f"{adj_views+sim_views}({adj_views}+{sim_views})"
        filter_name = config.get('filter', 'unknown')
        
        print(f"{i+1:2d}.  {status:<6} {name:<25} {total_views:<15} {filter_name:<15} {score:.6f}")
        
        if i < 3:
            medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
            print(f"     {medal} TOP {i+1}")
    
    print(f"\n🏆 BEST VIEW CONFIGURATION: {best_config['name']} - {best_score:.6f}")
    
    # Analysis
    print(f"\n📊 View Configuration Analysis:")
    
    # Adjacency vs Similarity analysis
    adj_only_scores = [score for name, config, score, success in results 
                      if config.get('use_adjacency_views', False) and not config.get('use_similarity_views', False) and success]
    sim_only_scores = [score for name, config, score, success in results 
                      if not config.get('use_adjacency_views', False) and config.get('use_similarity_views', False) and success]
    hybrid_scores = [score for name, config, score, success in results 
                    if config.get('use_adjacency_views', False) and config.get('use_similarity_views', False) and success]
    
    if adj_only_scores:
        print(f"  Adjacency-only average: {sum(adj_only_scores)/len(adj_only_scores):.6f}")
    if sim_only_scores:
        print(f"  Similarity-only average: {sum(sim_only_scores)/len(sim_only_scores):.6f}")
    if hybrid_scores:
        print(f"  Hybrid average: {sum(hybrid_scores)/len(hybrid_scores):.6f}")
    
    return best_config

def stage1_threshold_eigen_grid(best_view_config):
    """Stage 1: Threshold-Eigenvalue Grid Search for best view config"""
    print("\n" + "="*80)
    print("🎚️🧮 STAGE 1: Threshold-Eigenvalue Grid Search")
    print("="*80)
    
    # Only test thresholds if similarity views are used
    if best_view_config.get('use_similarity_views', False):
        threshold_values = [0.0, 0.001, 0.005, 0.01, 0.02]
    else:
        threshold_values = [0.0]  # No threshold needed for adjacency-only
    
    # Eigenvalue ranges
    u_values = [20, 25, 30, 35, 40]
    i_values = [30, 40, 50, 60, 70]
    
    print(f"Using best view config: {best_view_config['name']}")
    print(f"Testing {len(threshold_values)} thresholds × {len(u_values)} × {len(i_values)} eigenvalue combinations")
    
    best_score = 0
    best_config = None
    all_results = []
    
    total_combinations = len(threshold_values) * len(u_values) * len(i_values)
    combination_count = 0
    start_time = time.time()
    
    for threshold in threshold_values:
        print(f"\n🎚️ Testing threshold: {threshold:.3f}")
        
        for u_eigen in u_values:
            for i_eigen in i_values:
                combination_count += 1
                
                config = best_view_config.copy()
                config.update({
                    'epochs': 15,
                    'lr': 0.1,
                    'decay': 1e-4,
                    'similarity_threshold': threshold,
                    'u_n_eigen': u_eigen,
                    'i_n_eigen': i_eigen,
                    'verbose': 0,
                    'seed': 2025,
                    'model': 'multiview_uspec'
                })
                
                experiment_name = f"th={threshold:.3f}_u={u_eigen}_i={i_eigen}"
                print(f"[{combination_count:2d}/{total_combinations}] {experiment_name}")
                
                ndcg, success = run_experiment(config, experiment_name)
                
                result = {
                    'threshold': threshold,
                    'u_eigen': u_eigen,
                    'i_eigen': i_eigen,
                    'ndcg': ndcg,
                    'success': success,
                    'config': config
                }
                all_results.append(result)
                
                if success and ndcg > best_score:
                    best_score = ndcg
                    best_config = config.copy()
                    print(f"🎉 NEW BEST: {experiment_name} - {ndcg:.6f}")
                
                # Progress update
                elapsed = time.time() - start_time
                avg_time = elapsed / combination_count
                remaining = (total_combinations - combination_count) * avg_time
                print(f"Progress: {combination_count}/{total_combinations} ({combination_count/total_combinations*100:.1f}%) - ETA: {remaining/60:.1f}min")
    
    print(f"\n🏆 BEST THRESHOLD-EIGENVALUE CONFIG:")
    if best_config:
        print(f"  Threshold: {best_config['similarity_threshold']:.3f}")
        print(f"  U_Eigen: {best_config['u_n_eigen']}")
        print(f"  I_Eigen: {best_config['i_n_eigen']}")
        print(f"  NDCG@20: {best_score:.6f}")
    
    return best_config

def stage2_learning_rate(best_config):
    """Stage 2: Learning Rate Search"""
    print("\n" + "="*80)
    print("📈 STAGE 2: Learning Rate Search")
    print("="*80)
    
    lr_values = [0.2, 0.1, 0.05, 0.01]
    best_score = 0
    best_lr = None
    
    for lr in lr_values:
        config = best_config.copy()
        config.update({'epochs': 10, 'lr': lr, 'verbose': 0})
        
        ndcg, success = run_experiment(config, f"LR={lr}")
        
        if success and ndcg > best_score:
            best_score = ndcg
            best_lr = lr
    
    print(f"🏆 BEST LR: {best_lr} - {best_score:.6f}")
    return best_lr

def stage3_decay(best_config, best_lr):
    """Stage 3: Weight Decay Search"""
    print("\n" + "="*80)
    print("🎛️ STAGE 3: Weight Decay Search")
    print("="*80)
    
    decay_values = [1e-2, 1e-3, 1e-4, 1e-5]
    best_score = 0
    best_decay = None
    
    for decay in decay_values:
        config = best_config.copy()
        config.update({'epochs': 15, 'lr': best_lr, 'decay': decay, 'verbose': 0})
        
        ndcg, success = run_experiment(config, f"decay={decay}")
        
        if success and ndcg > best_score:
            best_score = ndcg
            best_decay = decay
    
    print(f"🏆 BEST DECAY: {best_decay} - {best_score:.6f}")
    return best_decay

def final_run(best_config, best_lr, best_decay):
    """Final run with optimal configuration"""
    print("\n" + "="*80)
    print("🏆 FINAL RUN WITH OPTIMAL MULTI-VIEW CONFIGURATION")
    print("="*80)
    
    final_config = best_config.copy()
    final_config.update({
        'epochs': 50,
        'lr': best_lr,
        'decay': best_decay,
        'verbose': 1,
        'seed': 2025
    })
    
    print(f"🎯 OPTIMAL MULTI-VIEW CONFIGURATION:")
    print(f"  View Configuration: {best_config['name']}")
    print(f"  Adjacency Views: {best_config.get('use_adjacency_views', False)}")
    print(f"  Similarity Views: {best_config.get('use_similarity_views', False)}")
    if best_config.get('use_adjacency_views', False):
        print(f"  Gamma Values: {best_config.get('gamma_values', [])}")
    if best_config.get('use_similarity_views', False):
        print(f"  Similarity Threshold: {best_config.get('similarity_threshold', 0.0):.3f}")
    print(f"  Filter Design: {best_config.get('filter', 'enhanced_basis')}")
    print(f"  User Eigenvalues: {best_config.get('u_n_eigen', 0)}")
    print(f"  Item Eigenvalues: {best_config.get('i_n_eigen', 0)}")
    print(f"  Learning Rate: {best_lr}")
    print(f"  Weight Decay: {best_decay}")
    print(f"  Training Epochs: 50 (full training)")
    
    # Calculate total views
    adj_views = len(best_config.get('gamma_values', [])) * 2 if best_config.get('use_adjacency_views', False) else 0
    sim_views = 2 if best_config.get('use_similarity_views', False) else 0
    total_views = adj_views + sim_views
    print(f"  Total Views: {total_views} ({adj_views} adjacency + {sim_views} similarity)")
    
    print(f"\n🚀 Running final evaluation...")
    ndcg, success = run_experiment(final_config, "FINAL OPTIMAL MULTI-VIEW")
    
    return ndcg, final_config

def save_results(search_results):
    """Save search results to file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"multiview_hp_search_results_ml100k_{timestamp}.json"
    
    # Convert any non-serializable objects
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(item) for item in obj]
        elif hasattr(obj, 'item'):  # numpy scalars
            return obj.item()
        else:
            return obj
    
    serializable_results = make_serializable(search_results)
    
    with open(filename, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"📁 Results saved to: {filename}")
    return filename

def full_multiview_search():
    """Run complete multi-view hierarchical search"""
    print("🔍 COMPLETE MULTI-VIEW HIERARCHICAL HYPERPARAMETER SEARCH FOR ML-100K")
    print("="*80)
    print("This will systematically optimize:")
    print("  Stage 0: Multi-view configurations (8 combinations)")
    print("  Stage 1: Threshold-eigenvalue grid for best view config")
    print("  Stage 2: Learning rates (4 values)")
    print("  Stage 3: Weight decay (4 values)")
    print("  Final: Optimal configuration with full training")
    print("="*80)
    
    total_start_time = time.time()
    
    # Stage 0: View configuration search
    stage0_start = time.time()
    best_view_config = stage0_view_configurations()
    stage0_time = time.time() - stage0_start
    
    # Stage 1: Threshold-eigenvalue grid
    stage1_start = time.time()
    best_config = stage1_threshold_eigen_grid(best_view_config)
    stage1_time = time.time() - stage1_start
    
    # Stage 2: Learning rate
    stage2_start = time.time()
    best_lr = stage2_learning_rate(best_config)
    stage2_time = time.time() - stage2_start
    
    # Stage 3: Weight decay
    stage3_start = time.time()
    best_decay = stage3_decay(best_config, best_lr)
    stage3_time = time.time() - stage3_start
    
    # Final run
    final_start = time.time()
    final_ndcg, final_config = final_run(best_config, best_lr, best_decay)
    final_time = time.time() - final_start
    
    total_time = time.time() - total_start_time
    
    # Final summary
    print(f"\n" + "="*80)
    print("📊 MULTI-VIEW HIERARCHICAL SEARCH COMPLETE")
    print("="*80)
    
    print(f"⏱️  Timing Summary:")
    print(f"  Stage 0 (View Configs): {stage0_time/60:.1f} minutes")
    print(f"  Stage 1 (Threshold-Eigenvalue Grid): {stage1_time/60:.1f} minutes")
    print(f"  Stage 2 (Learning Rate): {stage2_time/60:.1f} minutes")
    print(f"  Stage 3 (Weight Decay): {stage3_time/60:.1f} minutes")
    print(f"  Final Run: {final_time/60:.1f} minutes")
    print(f"  Total Time: {total_time/60:.1f} minutes ({total_time/3600:.1f} hours)")
    
    print(f"\n🏆 OPTIMAL MULTI-VIEW RESULTS:")
    print(f"  Final NDCG@20: {final_ndcg:.6f}")
    print(f"  View Configuration: {final_config['name']}")
    print(f"  Adjacency Views: {final_config.get('use_adjacency_views', False)}")
    print(f"  Similarity Views: {final_config.get('use_similarity_views', False)}")
    if final_config.get('use_adjacency_views', False):
        print(f"  Gamma Values: {final_config.get('gamma_values', [])}")
    if final_config.get('use_similarity_views', False):
        print(f"  Similarity Threshold: {final_config.get('similarity_threshold', 0.0):.3f}")
    print(f"  Filter Design: {final_config.get('filter', 'enhanced_basis')}")
    print(f"  Eigenvalues: u={final_config.get('u_n_eigen', 0)}, i={final_config.get('i_n_eigen', 0)}")
    print(f"  Learning Rate: {best_lr}")
    print(f"  Weight Decay: {best_decay}")
    
    # Calculate total views for final summary
    adj_views = len(final_config.get('gamma_values', [])) * 2 if final_config.get('use_adjacency_views', False) else 0
    sim_views = 2 if final_config.get('use_similarity_views', False) else 0
    total_views = adj_views + sim_views
    print(f"  Total Views: {total_views} ({adj_views} adjacency + {sim_views} similarity)")
    
    # Save results
    search_results = {
        'dataset': 'ml-100k',
        'search_type': 'multi_view_hierarchical',
        'timestamp': datetime.now().isoformat(),
        'total_time_minutes': total_time/60,
        'stage_times': {
            'stage0_view_configs': stage0_time/60,
            'stage1_threshold_eigen_grid': stage1_time/60,
            'stage2_lr': stage2_time/60,
            'stage3_decay': stage3_time/60,
            'final_run': final_time/60
        },
        'optimal_config': final_config,
        'final_ndcg': final_ndcg,
        'best_components': {
            'view_configuration': final_config['name'],
            'use_adjacency_views': final_config.get('use_adjacency_views', False),
            'use_similarity_views': final_config.get('use_similarity_views', False),
            'gamma_values': final_config.get('gamma_values', []),
            'similarity_threshold': final_config.get('similarity_threshold', 0.0),
            'filter_design': final_config.get('filter', 'enhanced_basis'),
            'u_n_eigen': final_config.get('u_n_eigen', 0),
            'i_n_eigen': final_config.get('i_n_eigen', 0),
            'learning_rate': best_lr,
            'weight_decay': best_decay,
            'total_views': total_views,
            'adjacency_views': adj_views,
            'similarity_views': sim_views
        }
    }
    
    results_file = save_results(search_results)
    
    print(f"\n🎯 Multi-view search completed! Check {results_file} for detailed results.")
    print("="*80)

def quick_view_test():
    """Quick test of different view configurations"""
    print("🚀 QUICK MULTI-VIEW TEST")
    print("="*40)
    
    view_configs = [
        {'name': 'similarity_only', 'use_adjacency_views': False, 'use_similarity_views': True, 'similarity_threshold': 0.01},
        {'name': 'adjacency_only', 'use_adjacency_views': True, 'use_similarity_views': False, 'gamma_values': [0.0, 0.5, 1.0]},
        {'name': 'hybrid', 'use_adjacency_views': True, 'use_similarity_views': True, 'gamma_values': [0.5], 'similarity_threshold': 0.01}
    ]
    
    for config in view_configs:
        print(f"\nTesting {config['name']}:")
        
        full_config = {
            'epochs': 5,
            'lr': 0.1,
            'decay': 1e-4,
            'u_n_eigen': 25,
            'i_n_eigen': 35,
            'filter': 'enhanced_basis',
            'verbose': 0,
            'seed': 2025,
            'model': 'multiview_uspec'
        }
        full_config.update(config)
        
        ndcg, success = run_experiment(full_config, f"quick_{config['name']}")

def test():
    """Test single multi-view experiment"""
    print("🧪 Testing single multi-view experiment...")
    
    config = {
        'epochs': 0,
        'lr': 0.1,
        'decay': 1e-4,
        'use_adjacency_views': True,
        'use_similarity_views': True,
        'gamma_values': [0.0, 0.5, 1.0],
        'similarity_threshold': 0.01,
        'u_n_eigen': 25,
        'i_n_eigen': 35,
        'filter': 'enhanced_basis',
        'init_filter': 'smooth',
        'verbose': 0,
        'seed': 2025,
        'model': 'multiview_uspec'
    }
    
    ndcg, success = run_experiment(config, "multi_view_test")
    if success:
        print(f"✅ Multi-view test successful! Ready for full search.")
    else:
        print(f"❌ Multi-view test failed. Check your configuration.")

if __name__ == "__main__":
    if len(sys.argv) > 1 and '--test' in sys.argv:
        test()
    elif len(sys.argv) > 1 and '--quick_test' in sys.argv:
        quick_view_test()
    elif len(sys.argv) > 1 and '--stage0' in sys.argv:
        stage0_view_configurations()
    else:
        full_multiview_search()