'''
Training Procedure with Detailed Progress Indicators
'''

import world
import numpy as np
import torch
import torch.nn as nn
import utils
from time import time
from tqdm import tqdm

class MSELoss:
    def __init__(self, model, config):
        print("🎯 Initializing optimizer...")
        
        self.model = model
        base_lr = config['lr']
        weight_decay = config['decay']
        
        # Setup optimizer with progress info
        try:
            filter_params = list(model.get_filter_parameters())
            other_params = list(model.get_other_parameters())
            
            if len(filter_params) > 0 and len(other_params) > 0:
                print(f"⚙️  Dual optimizer setup:")
                print(f"   ├─ Filter parameters: {len(filter_params)} groups, LR: {base_lr * 1.5:.4f}")
                print(f"   └─ Other parameters: {len(other_params)} groups, LR: {base_lr:.4f}")
                
                self.opt = torch.optim.Adam([
                    {'params': filter_params, 'lr': base_lr * 1.5, 'weight_decay': weight_decay * 0.1},
                    {'params': other_params, 'lr': base_lr, 'weight_decay': weight_decay}
                ])
            else:
                print(f"⚙️  Single optimizer setup: LR: {base_lr:.4f}")
                self.opt = torch.optim.Adam(model.parameters(), lr=base_lr, weight_decay=weight_decay)
        except:
            print(f"⚙️  Fallback optimizer setup: LR: {base_lr:.4f}")
            self.opt = torch.optim.Adam(model.parameters(), lr=base_lr, weight_decay=weight_decay)
        
        print("✅ Optimizer initialized successfully")
    
    def train_step(self, users, target_ratings):
        """Training step with loss tracking"""
        self.opt.zero_grad()
        predicted_ratings = self.model(users)
        
        # Ensure device consistency
        if predicted_ratings.device != target_ratings.device:
            target_ratings = target_ratings.to(predicted_ratings.device)
        
        # MSE loss
        loss = torch.mean((predicted_ratings - target_ratings) ** 2)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
        
        self.opt.step()
        return loss.item()

def create_target_ratings(dataset, users, device):
    """Create target ratings with progress for large batches"""
    batch_size = len(users)
    n_items = dataset.m_items
    
    # Create tensor on device directly
    target_ratings = torch.zeros(batch_size, n_items, device=device)
    
    # Show progress for large batches
    if batch_size > 1000:
        users_iter = tqdm(enumerate(users), total=len(users), 
                         desc="Creating targets", leave=False, disable=world.config.get('verbose', 1) == 0)
    else:
        users_iter = enumerate(users)
    
    # Vectorized assignment
    for i, user in users_iter:
        pos_items = dataset.allPos[user]
        if len(pos_items) > 0:
            target_ratings[i, pos_items] = 1.0
    
    return target_ratings

def train_epoch(dataset, model, loss_class, epoch, config):
    """Training epoch with detailed progress"""
    model.train()
    n_users = dataset.n_users
    train_batch_size = config['train_u_batch_size']
    
    # User sampling setup
    if train_batch_size == -1:
        users_per_epoch = n_users
        train_batch_size = n_users
        print(f"📊 Training on all {n_users:,} users")
    else:
        users_per_epoch = min(n_users, n_users // 2)
        print(f"📊 Training on {users_per_epoch:,}/{n_users:,} users, batch size: {train_batch_size:,}")
    
    # Sample users
    np.random.seed(epoch)
    sampled_users = np.random.choice(n_users, users_per_epoch, replace=False)
    
    # Training batches
    total_loss = 0.0
    n_batches = max(1, users_per_epoch // train_batch_size)
    
    # Progress bar for batches
    batch_iter = tqdm(range(n_batches), 
                     desc=f"Epoch {epoch+1} Batches", 
                     leave=False,
                     disable=config.get('verbose', 1) == 0)
    
    for batch_idx in batch_iter:
        start_idx = batch_idx * train_batch_size
        end_idx = min(start_idx + train_batch_size, users_per_epoch)
        user_indices = sampled_users[start_idx:end_idx]
        
        # Create tensors
        users = torch.LongTensor(user_indices).to(world.device)
        target_ratings = create_target_ratings(dataset, user_indices, world.device)
        
        # Training step
        batch_loss = loss_class.train_step(users, target_ratings)
        total_loss += batch_loss
        
        # Update progress bar with current loss
        batch_iter.set_postfix({'loss': f'{batch_loss:.6f}'})
    
    avg_loss = total_loss / n_batches
    return avg_loss

def evaluate(dataset, model, data_dict, config, eval_name="eval"):
    """Evaluation with progress tracking"""
    if len(data_dict) == 0:
        return {'recall': np.zeros(len(world.topks)),
                'precision': np.zeros(len(world.topks)),
                'ndcg': np.zeros(len(world.topks))}
    
    print(f"🔍 Starting {eval_name} evaluation...")
    
    model.eval()
    eval_batch_size = config['eval_u_batch_size']
    max_K = max(world.topks)
    
    results = {'recall': np.zeros(len(world.topks)),
               'precision': np.zeros(len(world.topks)),
               'ndcg': np.zeros(len(world.topks))}
    
    with torch.no_grad():
        users = list(data_dict.keys())
        n_eval_users = len(users)
        n_eval_batches = (n_eval_users + eval_batch_size - 1) // eval_batch_size
        
        print(f"📊 Evaluating {n_eval_users:,} users in {n_eval_batches} batches...")
        
        all_results = []
        
        # Progress bar for evaluation batches
        eval_iter = tqdm(utils.minibatch(users, batch_size=eval_batch_size), 
                        total=n_eval_batches,
                        desc=f"{eval_name.capitalize()} batches",
                        disable=config.get('verbose', 1) == 0)
        
        for batch_users in eval_iter:
            batch_users = [int(u) for u in batch_users]
            
            # Get data
            training_items = dataset.getUserPosItems(batch_users)
            ground_truth = [data_dict[u] for u in batch_users]
            
            # Get ratings
            batch_users_gpu = torch.LongTensor(batch_users).to(world.device)
            ratings = model.getUsersRating(batch_users_gpu)
            
            # Convert to tensor if needed
            if isinstance(ratings, np.ndarray):
                ratings = torch.from_numpy(ratings)
            else:
                ratings = ratings.cpu()
            
            # Mask training items
            for i, items in enumerate(training_items):
                if len(items) > 0:
                    ratings[i, items] = -float('inf')
            
            # Get top-K
            _, top_items = torch.topk(ratings, k=max_K)
            
            # Compute metrics
            batch_result = compute_metrics(ground_truth, top_items.numpy())
            all_results.append(batch_result)
            
            # Update progress with current NDCG
            if len(all_results) > 0:
                current_ndcg = np.mean([r['ndcg'][0] for r in all_results])
                eval_iter.set_postfix({'NDCG@20': f'{current_ndcg:.4f}'})
        
        # Aggregate results
        n_users = len(users)
        for result in all_results:
            results['recall'] += result['recall']
            results['precision'] += result['precision']
            results['ndcg'] += result['ndcg']
        
        # Average
        for key in results:
            results[key] /= n_users
    
    print(f"✅ {eval_name.capitalize()} evaluation complete")
    return results

def compute_metrics(ground_truth, predictions):
    """Compute metrics"""
    relevance = utils.getLabel(ground_truth, predictions)
    
    recall, precision, ndcg = [], [], []
    for k in world.topks:
        ret = utils.RecallPrecision_ATk(ground_truth, relevance, k)
        recall.append(ret['recall'])
        precision.append(ret['precision'])
        ndcg.append(utils.NDCGatK_r(ground_truth, relevance, k))
    
    return {'recall': np.array(recall),
            'precision': np.array(precision),
            'ndcg': np.array(ndcg)}

def train_and_evaluate(dataset, model, config):
    """Complete training pipeline with comprehensive progress tracking"""
    
    print("="*60)
    print(f"🚀 STARTING ENHANCED SPECTRAL CF TRAINING")
    print("="*60)
    
    # Setup info
    has_validation = hasattr(dataset, 'valDict') and len(dataset.valDict) > 0
    if has_validation:
        print(f"✅ Validation split available ({dataset.valDataSize:,} interactions)")
    else:
        print(f"⚠️  No validation split - using test data for early stopping")
    
    # Training configuration
    total_epochs = config['epochs']
    patience = config['patience']
    eval_every = config['n_epoch_eval']
    
    print(f"\n📋 Training Configuration:")
    print(f"   ├─ Total epochs: {total_epochs}")
    print(f"   ├─ Evaluation every: {eval_every} epochs")
    print(f"   ├─ Early stopping patience: {patience}")
    print(f"   ├─ Learning rate: {config['lr']:.4f}")
    print(f"   ├─ Weight decay: {config['decay']:.6f}")
    print(f"   └─ Device: {world.device}")
    
    # Initialize training components
    print(f"\n🔧 Initializing training components...")
    loss_class = MSELoss(model, config)
    
    # Training state
    best_ndcg = 0.0
    best_epoch = 0
    best_model_state = None
    no_improvement = 0
    
    print(f"\n📊 Dataset Summary:")
    print(f"   ├─ Training interactions: {dataset.trainDataSize:,}")
    print(f"   ├─ Validation interactions: {dataset.valDataSize:,}")
    print(f"   └─ Test users: {len(dataset.testDict):,}")
    
    # Ensure model is on device
    model = model.to(world.device)
    
    # Training loop with main progress bar
    print(f"\n🎯 Starting training loop...\n")
    start_time = time()
    
    # Main epoch progress bar
    epoch_iter = tqdm(range(total_epochs), 
                     desc="Training Progress",
                     unit="epoch")
    
    for epoch in epoch_iter:
        # Train one epoch
        epoch_start = time()
        avg_loss = train_epoch(dataset, model, loss_class, epoch, config)
        epoch_time = time() - epoch_start
        
        # Update epoch progress bar
        epoch_iter.set_postfix({
            'loss': f'{avg_loss:.6f}', 
            'best_ndcg': f'{best_ndcg:.4f}',
            'epoch_time': f'{epoch_time:.1f}s'
        })
        
        # Evaluation
        if (epoch + 1) % eval_every == 0 or epoch == total_epochs - 1:
            print(f"\n📊 Evaluation at epoch {epoch + 1}:")
            
            eval_data = dataset.valDict if has_validation else dataset.testDict
            eval_name = "validation" if has_validation else "test"
            
            results = evaluate(dataset, model, eval_data, config, eval_name)
            current_ndcg = results['ndcg'][0]
            
            # Check improvement
            if current_ndcg > best_ndcg:
                best_ndcg = current_ndcg
                best_epoch = epoch + 1
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improvement = 0
                
                print(f"🎉 NEW BEST! Epoch {epoch+1}:")
                print(f"   ├─ {eval_name.capitalize()} NDCG@20: {current_ndcg:.6f} ⭐")
                print(f"   ├─ Training loss: {avg_loss:.6f}")
                print(f"   └─ Epoch time: {epoch_time:.2f}s")
            else:
                no_improvement += 1
                print(f"📈 Epoch {epoch+1}:")
                print(f"   ├─ {eval_name.capitalize()} NDCG@20: {current_ndcg:.6f}")
                print(f"   ├─ Best NDCG@20: {best_ndcg:.6f} (epoch {best_epoch})")
                print(f"   ├─ Training loss: {avg_loss:.6f}")
                print(f"   ├─ No improvement for: {no_improvement} evaluations")
                print(f"   └─ Epoch time: {epoch_time:.2f}s")
            
            # Early stopping check
            if no_improvement >= patience:
                print(f"\n🛑 Early stopping triggered!")
                print(f"   └─ No improvement for {no_improvement} evaluations")
                epoch_iter.close()
                break
            
            print()  # Add spacing
    
    # Restore best model
    if best_model_state is not None:
        print(f"🔄 Restoring best model from epoch {best_epoch}...")
        model.load_state_dict(best_model_state)
        model = model.to(world.device)
        print(f"✅ Best model restored")
    
    # Final evaluation
    print(f"\n" + "="*60)
    print("🏆 FINAL TEST EVALUATION")
    print("="*60)
    
    final_results = evaluate(dataset, model, dataset.testDict, config, "final test")
    
    training_time = time() - start_time
    
    print(f"\n📊 Final Results Summary:")
    print(f"   ├─ Training time: {training_time:.2f}s")
    print(f"   ├─ Best epoch: {best_epoch}")
    print(f"   ├─ Recall@20: {final_results['recall'][0]:.6f}")
    print(f"   ├─ Precision@20: {final_results['precision'][0]:.6f}")
    print(f"   └─ NDCG@20: {final_results['ndcg'][0]:.6f}")
    
    print("="*60)
    
    return model, final_results