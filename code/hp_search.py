#!/usr/bin/env python3
"""
Complete Hierarchical Hyperparameter Search for ML-100K
Tests ALL filter designs, configurations, thresholds with corresponding eigenvalues
"""

import sys
import time
import json
from datetime import datetime
import numpy as np

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
        UniversalSpectralCF = MODELS['uspec']
        model = UniversalSpectralCF(adj_mat, world.config)
        
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

def stage0_filters():
    """Stage 0: Test ALL filter designs with ALL init filters (epochs=0)"""
    print("\n" + "="*80)
    print("🎯 STAGE 0: Complete Filter Analysis (NO TRAINING)")
    print("="*80)
    
    # ALL filter designs from your parse.py
    filter_designs = [
        'original', 'basis', 'enhanced_basis', 'adaptive_golden',
        'multiscale', 'ensemble', 'band_stop', 'adaptive_band_stop', 
        'parametric_multi_band', 'harmonic'
    ]
    
    # ALL init filters  
    init_filters = ['smooth', 'butterworth', 'gaussian', 'golden_036']
    
    best_score = 0
    best_filter = None
    results = []
    
    total_combinations = len(filter_designs) * len(init_filters)
    print(f"Testing {total_combinations} combinations ({len(filter_designs)} designs × {len(init_filters)} inits)")
    print("This reveals the inherent effectiveness of different filter initializations")
    
    count = 0
    start_time = time.time()
    
    for filter_design in filter_designs:
        for init_filter in init_filters:
            count += 1
            config = {
                'epochs': 0,
                'filter_design': filter_design,
                'init_filter': init_filter,
                'lr': 0.1,
                'decay': 1e-4,
                'similarity_threshold': 0.0,  # Default threshold for initial filter test
                'verbose': 0,
                'seed': 2025
            }
            
            print(f"\n[{count:2d}/{total_combinations}] Testing {filter_design}/{init_filter}...")
            ndcg, success = run_experiment(config, f"{filter_design}/{init_filter}")
            results.append((filter_design, init_filter, ndcg, success))
            
            if success and ndcg > best_score:
                best_score = ndcg
                best_filter = (filter_design, init_filter)
                print(f"🎉 NEW BEST: {filter_design}/{init_filter} - {ndcg:.6f}")
            
            # Progress update
            elapsed = time.time() - start_time
            avg_time = elapsed / count
            remaining = (total_combinations - count) * avg_time
            print(f"Progress: {count}/{total_combinations} ({count/total_combinations*100:.1f}%) - ETA: {remaining/60:.1f}min")
    
    # Show all results
    print(f"\n✅ Stage 0 Complete - All {total_combinations} Results:")
    print("="*100)
    
    # Sort by score
    results.sort(key=lambda x: x[2], reverse=True)
    
    print(f"{'Rank':<4} {'Status':<6} {'Filter Design':<20} {'Init Filter':<12} {'NDCG@20':<10}")
    print("-" * 100)
    
    for i, (design, init, score, success) in enumerate(results):
        status = "✅" if success else "❌"
        print(f"{i+1:2d}.  {status:<6} {design:<20} {init:<12} {score:.6f}")
        
        # Show top 3 prominently
        if i < 3:
            medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
            print(f"     {medal} TOP {i+1}")
    
    print(f"\n🏆 OVERALL BEST: {best_filter[0]}/{best_filter[1]} - {best_score:.6f}")
    
    return best_filter

def get_adaptive_eigen_counts(threshold):
    """Get adaptive eigenvalue counts based on threshold"""
    # Higher thresholds = sparser matrices = need fewer eigenvalues
    # Lower thresholds = denser matrices = can use more eigenvalues
    
    if threshold >= 0.05:
        # High threshold: very sparse, use fewer eigenvalues
        u_values = [10, 15, 20, 25]
        i_values = [15, 25, 35, 45]
    elif threshold >= 0.01:
        # Medium-high threshold: moderate sparsity
        u_values = [15, 20, 25, 30, 35]
        i_values = [25, 35, 45, 55, 65]
    elif threshold >= 0.005:
        # Medium threshold: balanced sparsity
        u_values = [20, 25, 30, 35, 40]
        i_values = [30, 40, 50, 60, 70]
    elif threshold >= 0.001:
        # Low threshold: less sparse
        u_values = [25, 30, 35, 40, 45]
        i_values = [35, 45, 55, 65, 75]
    else:
        # No threshold or very low: can use more eigenvalues
        u_values = [30, 35, 40, 45, 50]
        i_values = [40, 50, 60, 70, 80]
    
    return u_values, i_values

def stage1_threshold_eigen_grid_search(best_filter):
    """Stage 1: Combined Threshold-Eigenvalue Grid Search"""
    print("\n" + "="*80)
    print("🎚️🧮 STAGE 1: Combined Threshold-Eigenvalue Grid Search")
    print("="*80)
    
    # Threshold values for ML-100K
    threshold_values = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05]
    
    print(f"Using best filter from Stage 0: {best_filter[0]}/{best_filter[1]}")
    print("Testing threshold-eigenvalue combinations with adaptive eigenvalue ranges")
    
    best_score = 0
    best_config = None
    all_results = []
    
    total_combinations = 0
    # Calculate total combinations
    for threshold in threshold_values:
        u_values, i_values = get_adaptive_eigen_counts(threshold)
        total_combinations += len(u_values) * len(i_values)
    
    print(f"Total combinations to test: {total_combinations}")
    print("Eigenvalue ranges adapt to threshold sparsity:")
    for threshold in threshold_values:
        u_values, i_values = get_adaptive_eigen_counts(threshold)
        print(f"  Threshold {threshold:.3f}: u_eigen={u_values}, i_eigen={i_values}")
    
    combination_count = 0
    start_time = time.time()
    
    for threshold_idx, threshold in enumerate(threshold_values):
        print(f"\n" + "-"*60)
        print(f"🎚️ Testing Threshold: {threshold:.3f} ({threshold_idx+1}/{len(threshold_values)})")
        print("-"*60)
        
        u_values, i_values = get_adaptive_eigen_counts(threshold)
        threshold_results = []
        best_threshold_score = 0
        best_threshold_config = None
        
        for u_eigen in u_values:
            for i_eigen in i_values:
                combination_count += 1
                
                config = {
                    'epochs': 15,  # Medium training for grid search
                    'lr': 0.1,
                    'decay': 1e-4,
                    'similarity_threshold': threshold,
                    'u_n_eigen': u_eigen,
                    'i_n_eigen': i_eigen,
                    'filter_design': best_filter[0],
                    'init_filter': best_filter[1],
                    'verbose': 0,
                    'seed': 2025
                }
                
                experiment_name = f"th={threshold:.3f}_u={u_eigen}_i={i_eigen}"
                print(f"\n[{combination_count:2d}/{total_combinations}] {experiment_name}")
                
                ndcg, success = run_experiment(config, experiment_name)
                
                result = {
                    'threshold': threshold,
                    'u_eigen': u_eigen,
                    'i_eigen': i_eigen,
                    'ndcg': ndcg,
                    'success': success,
                    'config': config
                }
                
                threshold_results.append(result)
                all_results.append(result)
                
                if success and ndcg > best_score:
                    best_score = ndcg
                    best_config = config.copy()
                    print(f"🎉 NEW GLOBAL BEST: {experiment_name} - {ndcg:.6f}")
                
                if success and ndcg > best_threshold_score:
                    best_threshold_score = ndcg
                    best_threshold_config = config.copy()
                
                # Progress update
                elapsed = time.time() - start_time
                avg_time = elapsed / combination_count
                remaining = (total_combinations - combination_count) * avg_time
                print(f"Progress: {combination_count}/{total_combinations} ({combination_count/total_combinations*100:.1f}%) - ETA: {remaining/60:.1f}min")
        
        # Show best for this threshold
        if best_threshold_config:
            print(f"\n✅ Best for threshold {threshold:.3f}:")
            print(f"   u_eigen={best_threshold_config['u_n_eigen']}, i_eigen={best_threshold_config['i_n_eigen']}")
            print(f"   NDCG@20: {best_threshold_score:.6f}")
    
    # Comprehensive analysis
    print(f"\n" + "="*80)
    print("📊 COMPREHENSIVE THRESHOLD-EIGENVALUE ANALYSIS")
    print("="*80)
    
    # Best results per threshold
    print("\n🏆 Best Configuration per Threshold:")
    print(f"{'Threshold':<10} {'U_Eigen':<8} {'I_Eigen':<8} {'NDCG@20':<10} {'Improvement':<12}")
    print("-" * 60)
    
    threshold_best = {}
    for threshold in threshold_values:
        threshold_results = [r for r in all_results if r['threshold'] == threshold and r['success']]
        if threshold_results:
            best_for_threshold = max(threshold_results, key=lambda x: x['ndcg'])
            threshold_best[threshold] = best_for_threshold
            
            baseline_score = threshold_best.get(0.0, {}).get('ndcg', 0) if 0.0 in threshold_best else 0
            improvement = best_for_threshold['ndcg'] - baseline_score if baseline_score > 0 else 0
            improvement_str = f"{improvement:+.4f}" if baseline_score > 0 else "N/A"
            
            print(f"{threshold:<10.3f} {best_for_threshold['u_eigen']:<8} {best_for_threshold['i_eigen']:<8} "
                  f"{best_for_threshold['ndcg']:<10.6f} {improvement_str:<12}")
    
    # Eigenvalue trends analysis
    print(f"\n📈 Eigenvalue Trends by Threshold:")
    for threshold in sorted(threshold_best.keys()):
        if threshold in threshold_best:
            result = threshold_best[threshold]
            ratio = result['i_eigen'] / result['u_eigen'] if result['u_eigen'] > 0 else 0
            print(f"  Threshold {threshold:.3f}: u={result['u_eigen']}, i={result['i_eigen']}, "
                  f"ratio={ratio:.2f}, NDCG={result['ndcg']:.6f}")
    
    # Global best
    print(f"\n🏆 GLOBAL BEST CONFIGURATION:")
    if best_config:
        print(f"  Threshold: {best_config['similarity_threshold']:.3f}")
        print(f"  U_Eigen: {best_config['u_n_eigen']}")
        print(f"  I_Eigen: {best_config['i_n_eigen']}")
        print(f"  NDCG@20: {best_score:.6f}")
        print(f"  I/U Ratio: {best_config['i_n_eigen']/best_config['u_n_eigen']:.2f}")
    
    return best_config

def stage2_learning_rate(best_config):
    """Stage 2: Learning Rate Search with optimal threshold-eigenvalue config"""
    print("\n" + "="*80)
    print("📈 STAGE 2: Learning Rate Search")
    print("="*80)
    
    lr_values = [0.2, 0.1, 0.05, 0.01]
    best_score = 0
    best_lr = None
    results = []
    
    print(f"Using optimal config: threshold={best_config['similarity_threshold']:.3f}, "
          f"u_eigen={best_config['u_n_eigen']}, i_eigen={best_config['i_n_eigen']}")
    print(f"Testing {len(lr_values)} learning rates with short training (10 epochs)")
    
    for i, lr in enumerate(lr_values):
        config = best_config.copy()
        config.update({
            'epochs': 10,
            'lr': lr,
            'verbose': 0,
            'seed': 2025
        })
        
        print(f"\n[{i+1}/{len(lr_values)}] Testing LR={lr}...")
        ndcg, success = run_experiment(config, f"LR={lr}")
        results.append((lr, ndcg, success))
        
        if success and ndcg > best_score:
            best_score = ndcg
            best_lr = lr
            print(f"🎉 NEW BEST LR: {lr} - {ndcg:.6f}")
    
    print(f"\n✅ Stage 2 Results:")
    print(f"{'Rank':<4} {'Status':<6} {'Learning Rate':<12} {'NDCG@20':<10}")
    print("-" * 50)
    
    results.sort(key=lambda x: x[1], reverse=True)
    for i, (lr, score, success) in enumerate(results):
        status = "✅" if success else "❌"
        print(f"{i+1:2d}.  {status:<6} {lr:<12} {score:.6f}")
    
    print(f"\n🏆 BEST LR: {best_lr} - {best_score:.6f}")
    
    return best_lr

def stage3_decay(best_config, best_lr):
    """Stage 3: Weight Decay Search"""
    print("\n" + "="*80)
    print("🎛️ STAGE 3: Weight Decay Search") 
    print("="*80)
    
    decay_values = [1e-2, 1e-3, 1e-4, 1e-5]
    best_score = 0
    best_decay = None
    results = []
    
    print(f"Using optimal config with LR={best_lr}")
    print(f"Testing {len(decay_values)} weight decay values with medium training (15 epochs)")
    
    for i, decay in enumerate(decay_values):
        config = best_config.copy()
        config.update({
            'epochs': 15,
            'lr': best_lr,
            'decay': decay,
            'verbose': 0,
            'seed': 2025
        })
        
        print(f"\n[{i+1}/{len(decay_values)}] Testing decay={decay}...")
        ndcg, success = run_experiment(config, f"decay={decay}")
        results.append((decay, ndcg, success))
        
        if success and ndcg > best_score:
            best_score = ndcg
            best_decay = decay
            print(f"🎉 NEW BEST DECAY: {decay} - {ndcg:.6f}")
    
    print(f"\n✅ Stage 3 Results:")
    print(f"{'Rank':<4} {'Status':<6} {'Weight Decay':<12} {'NDCG@20':<10}")
    print("-" * 50)
    
    results.sort(key=lambda x: x[1], reverse=True)
    for i, (decay, score, success) in enumerate(results):
        status = "✅" if success else "❌"
        print(f"{i+1:2d}.  {status:<6} {decay:<12} {score:.6f}")
    
    print(f"\n🏆 BEST DECAY: {best_decay} - {best_score:.6f}")
    
    return best_decay

def final_run(best_config, best_lr, best_decay):
    """Final run with best configuration"""
    print("\n" + "="*80)
    print("🏆 FINAL RUN WITH OPTIMAL CONFIGURATION")
    print("="*80)
    
    final_config = best_config.copy()
    final_config.update({
        'epochs': 50,
        'lr': best_lr,
        'decay': best_decay,
        'verbose': 1,
        'seed': 2025
    })
    
    print(f"🎯 OPTIMAL CONFIGURATION:")
    print(f"  Filter Design: {final_config['filter_design']}")
    print(f"  Init Filter: {final_config['init_filter']}")
    print(f"  Similarity Threshold: {final_config['similarity_threshold']:.3f}")
    print(f"  User Eigenvalues (u_n_eigen): {final_config['u_n_eigen']}")
    print(f"  Item Eigenvalues (i_n_eigen): {final_config['i_n_eigen']}")
    print(f"  I/U Eigenvalue Ratio: {final_config['i_n_eigen']/final_config['u_n_eigen']:.2f}")
    print(f"  Learning Rate: {best_lr}")
    print(f"  Weight Decay: {best_decay}")
    print(f"  Training Epochs: 50 (full training)")
    
    print(f"\n🚀 Running final evaluation...")
    ndcg, success = run_experiment(final_config, "FINAL OPTIMAL")
    
    return ndcg, final_config

def save_results(search_results):
    """Save search results to file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"hp_search_results_ml100k_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(search_results, f, indent=2)
    
    print(f"📁 Results saved to: {filename}")
    return filename

def full_search():
    """Run complete hierarchical search with threshold-eigenvalue grid"""
    print("🔍 COMPLETE HIERARCHICAL HYPERPARAMETER SEARCH FOR ML-100K")
    print("="*80)
    print("This will systematically optimize:")
    print("  Stage 0: Filter designs + init filters (40 combinations)")
    print("  Stage 1: Threshold-Eigenvalue Grid Search (~100-150 combinations)")
    print("  Stage 2: Learning rates (4 values)")
    print("  Stage 3: Weight decay (4 values)")
    print("  Final: Optimal configuration with full training")
    print("="*80)
    
    total_start_time = time.time()
    
    # Stage 0: Filter analysis
    stage0_start = time.time()
    best_filter = stage0_filters()
    stage0_time = time.time() - stage0_start
    
    # Stage 1: Threshold-Eigenvalue Grid Search
    stage1_start = time.time()
    best_config = stage1_threshold_eigen_grid_search(best_filter)
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
    print("📊 HIERARCHICAL SEARCH COMPLETE")
    print("="*80)
    
    print(f"⏱️  Timing Summary:")
    print(f"  Stage 0 (Filters): {stage0_time/60:.1f} minutes")
    print(f"  Stage 1 (Threshold-Eigenvalue Grid): {stage1_time/60:.1f} minutes")
    print(f"  Stage 2 (Learning Rate): {stage2_time/60:.1f} minutes")
    print(f"  Stage 3 (Weight Decay): {stage3_time/60:.1f} minutes") 
    print(f"  Final Run: {final_time/60:.1f} minutes")
    print(f"  Total Time: {total_time/60:.1f} minutes ({total_time/3600:.1f} hours)")
    
    print(f"\n🏆 OPTIMAL RESULTS:")
    print(f"  Final NDCG@20: {final_ndcg:.6f}")
    print(f"  Filter: {final_config['filter_design']}/{final_config['init_filter']}")
    print(f"  Similarity Threshold: {final_config['similarity_threshold']:.3f}")
    print(f"  Eigenvalues: u={final_config['u_n_eigen']}, i={final_config['i_n_eigen']}")
    print(f"  I/U Ratio: {final_config['i_n_eigen']/final_config['u_n_eigen']:.2f}")
    print(f"  Learning Rate: {best_lr}")
    print(f"  Weight Decay: {best_decay}")
    
    # Save results
    search_results = {
        'dataset': 'ml-100k',
        'timestamp': datetime.now().isoformat(),
        'total_time_minutes': total_time/60,
        'stage_times': {
            'stage0_filters': stage0_time/60,
            'stage1_threshold_eigen_grid': stage1_time/60,
            'stage2_lr': stage2_time/60,
            'stage3_decay': stage3_time/60,
            'final_run': final_time/60
        },
        'optimal_config': final_config,
        'final_ndcg': final_ndcg,
        'best_components': {
            'filter_design': final_config['filter_design'],
            'init_filter': final_config['init_filter'],
            'similarity_threshold': final_config['similarity_threshold'],
            'u_n_eigen': final_config['u_n_eigen'],
            'i_n_eigen': final_config['i_n_eigen'],
            'eigenvalue_ratio': final_config['i_n_eigen']/final_config['u_n_eigen'],
            'learning_rate': best_lr,
            'weight_decay': best_decay
        }
    }
    
    results_file = save_results(search_results)
    
    print(f"\n🎯 Search completed! Check {results_file} for detailed results.")
    print("="*80)

def quick_threshold_eigen_test():
    """Quick test of threshold-eigenvalue relationship"""
    print("🚀 QUICK THRESHOLD-EIGENVALUE TEST")
    print("="*40)
    
    best_filter = ('enhanced_basis', 'smooth')  # Use known good filter
    threshold_values = [0.0, 0.01, 0.05]
    
    for threshold in threshold_values:
        print(f"\nTesting threshold {threshold:.3f}:")
        u_values, i_values = get_adaptive_eigen_counts(threshold)
        print(f"  Adaptive ranges: u_eigen={u_values}, i_eigen={i_values}")
        
        # Test just the middle values
        u_eigen = u_values[len(u_values)//2]
        i_eigen = i_values[len(i_values)//2]
        
        config = {
            'epochs': 5,
            'lr': 0.1,
            'decay': 1e-4,
            'similarity_threshold': threshold,
            'u_n_eigen': u_eigen,
            'i_n_eigen': i_eigen,
            'filter_design': best_filter[0],
            'init_filter': best_filter[1],
            'verbose': 0,
            'seed': 2025
        }
        
        ndcg, success = run_experiment(config, f"quick_th={threshold:.3f}")

def test():
    """Test single experiment"""
    print("🧪 Testing single experiment...")
    
    config = {
        'epochs': 0,
        'lr': 0.1,
        'decay': 1e-4,
        'similarity_threshold': 0.01,
        'u_n_eigen': 25,
        'i_n_eigen': 35,
        'filter_design': 'enhanced_basis',
        'init_filter': 'smooth',
        'verbose': 0,
        'seed': 2025
    }
    
    ndcg, success = run_experiment(config, "single_test")
    if success:
        print(f"✅ Test successful! Ready for full search.")
    else:
        print(f"❌ Test failed. Check your configuration.")

if __name__ == "__main__":
    if len(sys.argv) > 1 and '--test' in sys.argv:
        test()
    elif len(sys.argv) > 1 and '--stage0' in sys.argv:
        stage0_filters()
    elif len(sys.argv) > 1 and '--quick_test' in sys.argv:
        quick_threshold_eigen_test()
    else:
        full_search()