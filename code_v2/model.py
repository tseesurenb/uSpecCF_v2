'''
Created on June 7, 2025
PyTorch Implementation of uSpec: Universal Spectral Collaborative Filtering - SVD Optimized

@author: Tseesuren Batsuuri (tseesuren.batsuuri@hdr.mq.edu.au)
'''

import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from scipy.linalg import svd as scipy_svd
import time
import gc
import os
import pickle
import filters as fl
from tqdm import tqdm


class UniversalSpectralCF(nn.Module):
    """Universal Spectral CF with SVD Optimization"""
    
    def __init__(self, adj_mat, config=None):
        super().__init__()
        
        print("🚀 Initializing Universal Spectral CF with SVD optimization...")
        
        self.config = config or {}
        self.device = self.config.get('device', 'cpu')
        self.filter_order = self.config.get('filter_order', 6)
        self.in_mat = self.config.get('in_mat', 'ui')
        
        print("📊 Processing adjacency matrix...")
        if sp.issparse(adj_mat):
            adj_dense = adj_mat.toarray()
        else:
            adj_dense = adj_mat
            
        self.register_buffer('adj_tensor', torch.tensor(adj_dense, dtype=torch.float32))
        self.n_users, self.n_items = self.adj_tensor.shape
        
        # Calculate basic statistics
        total_interactions = int(torch.sum(self.adj_tensor).item())
        sparsity = total_interactions / (self.n_users * self.n_items)
                
        # SVD configuration (using eigenvalue terminology for consistency)
        self.u_n_components, self.i_n_components = self._get_svd_component_counts(total_interactions, sparsity)
        self.similarity_threshold = self.config.get('similarity_threshold', 0.0)
        self.filter = self.config.get('filter', 'enhanced_basis')
        self.init_filter = self.config.get('init_filter', 'smooth')
        
        # Clean up
        del adj_dense
        gc.collect()
        
        # Setup components with progress
        print("\n🔧 Setting up spectral filters with SVD...")
        self._setup_filters()
        
        print("🎯 Initializing combination weights...")
        self._setup_combination_weights()
        
        print("✅ Model initialization complete!\n")
    
    def _get_svd_component_counts(self, total_interactions, sparsity):
        """Calculate SVD component counts (equivalent to eigenvalue counts)"""
        manual_u = self.config.get('u_n_eigen', 0)
        manual_i = self.config.get('i_n_eigen', 0)
        
        if manual_u > 0 and manual_i > 0:
            print(f"🎯 Using manual SVD component counts: u={manual_u}, i={manual_i}")
            return manual_u, manual_i
        
        print("🤖 Computing adaptive SVD component counts...")
        
        # Simplified adaptive calculation
        base_u = min(max(16, self.n_users // 20), 128)
        base_i = min(max(16, self.n_items // 20), 128)
        
        # Sparsity adjustment
        if sparsity < 0.01:
            multiplier = 1.2
            print(f"   └─ Sparse dataset detected, increasing components by 20%")
        elif sparsity > 0.05:
            multiplier = 0.8
            print(f"   └─ Dense dataset detected, reducing components by 20%")
        else:
            multiplier = 1.0
            print(f"   └─ Normal sparsity, using standard component counts")
            
        u_components = int(base_u * multiplier)
        i_components = int(base_i * multiplier)
        
        print(f"   ├─ User SVD components: {base_u} → {u_components}")
        print(f"   └─ Item SVD components: {base_i} → {i_components}")
        
        return u_components, i_components
    
    def _get_cache_path(self, cache_type, filter_type=None):
        """Generate cache path"""
        cache_dir = "../cache"
        os.makedirs(cache_dir, exist_ok=True)
        
        dataset = self.config.get('dataset', 'unknown')
        threshold = str(self.similarity_threshold).replace('.', 'p')
        
        if filter_type:
            k_value = self.u_n_components if filter_type == 'user' else self.i_n_components
            filename = f"{dataset}_svd_{filter_type}_{cache_type}_k{k_value}_th{threshold}.pkl"
        else:
            filename = f"{dataset}_svd_{cache_type}_th{threshold}.pkl"
            
        return os.path.join(cache_dir, filename)
    
    def _compute_similarity_matrix(self, interaction_matrix, cache_type=None):
        """Compute similarity matrix with progress"""
        
        print(f"    🔍 Computing {cache_type} similarity matrix...")
        
        # Check cache first
        if cache_type:
            cache_path = self._get_cache_path('sim', cache_type)
            if os.path.exists(cache_path):
                try:
                    print(f"    📂 Loading from cache: {os.path.basename(cache_path)}")
                    with open(cache_path, 'rb') as f:
                        similarity = pickle.load(f)
                    print(f"    ✅ Cache loaded successfully")
                    return similarity.to(self.device)
                except Exception as e:
                    print(f"    ⚠️  Cache loading failed: {e}")
        
        # Compute A @ A.T
        print(f"    ⚡ Computing matrix multiplication A @ A.T...")
        start_time = time.time()
        
        with torch.no_grad():
            similarity = torch.mm(interaction_matrix, interaction_matrix.t())
            compute_time = time.time() - start_time
            print(f"    ⏱️  Matrix multiplication completed in {compute_time:.2f}s")
            
            # Apply threshold
            if self.similarity_threshold > 0:
                print(f"    🎯 Applying similarity threshold: {self.similarity_threshold}")
                before_nonzero = (similarity > 0).float().mean().item()
                
                similarity = torch.where(similarity >= self.similarity_threshold, 
                                       similarity, torch.zeros_like(similarity))
                
                after_nonzero = (similarity > 0).float().mean().item()
                print(f"    📊 Similarity sparsity: {before_nonzero:.3f} → {after_nonzero:.3f}")
            
            # Set diagonal and clamp
            similarity.fill_diagonal_(1.0)
            similarity = torch.clamp(similarity, min=0.0, max=1.0)
        
        # Save to cache
        if cache_type:
            try:
                print(f"    💾 Saving to cache...")
                with open(cache_path, 'wb') as f:
                    pickle.dump(similarity.cpu(), f)
                print(f"    ✅ Cache saved successfully")
            except Exception as e:
                print(f"    ⚠️  Cache saving failed: {e}")
        
        return similarity
    
    def _compute_similarity_laplacian(self, similarity_matrix):
        """Compute Laplacian with progress"""
        print(f"    🔧 Computing normalized similarity Laplacian...")
        
        # Compute degree vector
        degree = similarity_matrix.sum(dim=1) + 1e-8
        
        # Symmetric normalization
        deg_inv_sqrt = torch.pow(degree, -0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
        
        # Efficient normalization
        normalized = similarity_matrix * deg_inv_sqrt.unsqueeze(0)
        normalized = normalized * deg_inv_sqrt.unsqueeze(1)
        
        # L = I - normalized
        identity = torch.eye(similarity_matrix.shape[0], device=similarity_matrix.device)
        laplacian = identity - normalized
        
        print(f"    ✅ Laplacian computation completed")
        return laplacian
    
    def _setup_filters(self):
        """Setup filters with detailed progress"""
        self.user_filter = None
        self.item_filter = None
        
        if self.in_mat in ['u', 'ui']:
            print(f"👤 Setting up user similarity filter...")
            self.user_filter = self._create_filter('user')
            print(f"✅ User filter setup complete")
        
        if self.in_mat in ['i', 'ui']:
            print(f"🎬 Setting up item similarity filter...")
            self.item_filter = self._create_filter('item')
            print(f"✅ Item filter setup complete")
    
    def _create_filter(self, filter_type):
        """Create filter with detailed progress using SVD"""
        n_components = self.u_n_components if filter_type == 'user' else self.i_n_components
        n_matrix_size = self.n_users if filter_type == 'user' else self.n_items
        
        print(f"  📋 {filter_type.capitalize()} filter configuration:")
        print(f"     ├─ Matrix size: {n_matrix_size:,}")
        print(f"     └─ SVD components: {n_components}")
        
        # Try cache first
        cache_path = self._get_cache_path('svd', filter_type)
        if os.path.exists(cache_path):
            try:
                print(f"  📂 Loading SVD decomposition from cache...")
                with open(cache_path, 'rb') as f:
                    singular_vals, svd_vectors = pickle.load(f)
                self.register_buffer(f'{filter_type}_singular_vals', singular_vals.to(self.device))
                self.register_buffer(f'{filter_type}_svd_vectors', svd_vectors.to(self.device))
                print(f"  ✅ SVD decomposition loaded from cache")
                return self._create_filter_instance()
            except Exception as e:
                print(f"  ⚠️  Cache loading failed: {e}")
        
        # Compute SVD decomposition
        print(f"  🔄 Computing SVD decomposition...")
        start_time = time.time()
        
        with torch.no_grad():
            if filter_type == 'user':
                similarity = self._compute_similarity_matrix(self.adj_tensor, 'user')
            else:
                similarity = self._compute_similarity_matrix(self.adj_tensor.t(), 'item')
            
            laplacian = self._compute_similarity_laplacian(similarity)
        
        # SVD decomposition with progress
        print(f"  ⚡ Running SVD decomposition (much faster than eigendecomposition)...")
        svd_start = time.time()
        
        laplacian_np = laplacian.cpu().numpy()
        k = min(n_components, n_matrix_size - 2)
        
        try:
            print(f"  🎯 Computing {k} components with SVD...")
            
            # Ensure matrix is symmetric for spectral properties
            laplacian_np = (laplacian_np + laplacian_np.T) / 2
            
            # Add small regularization
            np.fill_diagonal(laplacian_np, laplacian_np.diagonal() + 1e-6)
            
            # Use TruncatedSVD for efficiency
            if k < n_matrix_size - 1:
                svd_solver = TruncatedSVD(n_components=k, random_state=42)
                svd_solver.fit(laplacian_np)
                
                singular_vals = svd_solver.singular_values_
                svd_vectors = svd_solver.components_.T
            else:
                # For small matrices, use full SVD
                U, s, Vt = scipy_svd(laplacian_np, full_matrices=False)
                singular_vals = s[:k]
                svd_vectors = U[:, :k]
            
            # Convert to "eigenvalue-like" values for spectral filtering
            singular_vals = np.maximum(singular_vals, 0.0)
            
            svd_time = time.time() - svd_start
            print(f"  ⏱️  SVD decomposition completed in {svd_time:.2f}s")
            print(f"  📊 Singular value range: [{singular_vals.min():.4f}, {singular_vals.max():.4f}]")
            
            singular_vals_tensor = torch.tensor(singular_vals, dtype=torch.float32)
            svd_vectors_tensor = torch.tensor(svd_vectors, dtype=torch.float32)
            
            # Save to cache
            try:
                print(f"  💾 Saving SVD decomposition to cache...")
                with open(cache_path, 'wb') as f:
                    pickle.dump((singular_vals_tensor, svd_vectors_tensor), f)
                print(f"  ✅ Cache saved successfully")
            except Exception as e:
                print(f"  ⚠️  Cache saving failed: {e}")
            
            self.register_buffer(f'{filter_type}_singular_vals', singular_vals_tensor.to(self.device))
            self.register_buffer(f'{filter_type}_svd_vectors', svd_vectors_tensor.to(self.device))
            
        except Exception as e:
            print(f"  ❌ SVD decomposition failed: {e}")
            print(f"  🔄 Using fallback identity matrices...")
            
            singular_vals = np.linspace(0.01, 1, min(n_components, n_matrix_size))
            singular_vals_tensor = torch.tensor(singular_vals, dtype=torch.float32)
            svd_vectors_tensor = torch.eye(n_matrix_size, min(n_components, n_matrix_size))
            
            self.register_buffer(f'{filter_type}_singular_vals', singular_vals_tensor.to(self.device))
            self.register_buffer(f'{filter_type}_svd_vectors', svd_vectors_tensor.to(self.device))
        
        total_time = time.time() - start_time
        print(f"  ⏱️  Total {filter_type} filter setup: {total_time:.2f}s")
        
        # Clean up
        del laplacian_np, similarity, laplacian
        gc.collect()
        
        return self._create_filter_instance()
    
    def _create_filter_instance(self):
        """Create filter instance"""
        if self.filter == 'enhanced_basis':
            return fl.EnhancedSpectralBasisFilter(self.filter_order, self.init_filter)
        elif self.filter == 'multiscale':
            return fl.MultiScaleSpectralFilter(self.filter_order, self.init_filter)
        elif self.filter == 'ensemble':
            return fl.EnsembleSpectralFilter(self.filter_order, self.init_filter)
        else:
            return fl.UniversalSpectralFilter(self.filter_order, self.init_filter)
    
    def _setup_combination_weights(self):
        """Setup combination weights"""
        if self.in_mat == 'ui':
            weights = torch.tensor([0.5, 0.3, 0.2])
            print(f"  🎚️  UI filter weights: {weights.tolist()}")
        else:
            weights = torch.tensor([0.5, 0.5])
            print(f"  🎚️  Single filter weights: {weights.tolist()}")
        
        self.combination_weights = nn.Parameter(weights.to(self.device))
    
    def _get_filter_matrices(self):
        """Compute filter matrices using SVD data"""
        user_matrix = item_matrix = None
        
        if self.user_filter is not None:
            response = self.user_filter(self.user_singular_vals)
            user_matrix = torch.mm(torch.mm(self.user_svd_vectors, torch.diag(response)), 
                                 self.user_svd_vectors.t())
        
        if self.item_filter is not None:
            response = self.item_filter(self.item_singular_vals)
            item_matrix = torch.mm(torch.mm(self.item_svd_vectors, torch.diag(response)), 
                                 self.item_svd_vectors.t())
        
        return user_matrix, item_matrix
    
    def forward(self, users):
        """Forward pass"""
        if users.device != self.adj_tensor.device:
            users = users.to(self.adj_tensor.device)
        
        user_profiles = self.adj_tensor[users]
        user_filter_matrix, item_filter_matrix = self._get_filter_matrices()
        
        scores = [user_profiles]
        
        if self.in_mat in ['i', 'ui'] and item_filter_matrix is not None:
            scores.append(torch.mm(user_profiles, item_filter_matrix))
        
        if self.in_mat in ['u', 'ui'] and user_filter_matrix is not None:
            scores.append(torch.mm(user_filter_matrix[users], self.adj_tensor))
        
        weights = torch.softmax(self.combination_weights, dim=0)
        predicted = sum(w * score for w, score in zip(weights, scores))
        
        return predicted
    
    def getUsersRating(self, batch_users):
        """Evaluation interface"""
        self.eval()
        with torch.no_grad():
            if isinstance(batch_users, np.ndarray):
                batch_users = torch.LongTensor(batch_users)
            
            if batch_users.device != self.device:
                batch_users = batch_users.to(self.device)
            
            return self.forward(batch_users).cpu().numpy()
    
    def get_filter_parameters(self):
        """Get filter parameters"""
        params = []
        if self.user_filter is not None:
            params.extend(self.user_filter.parameters())
        if self.item_filter is not None:
            params.extend(self.item_filter.parameters())
        return params
    
    def get_other_parameters(self):
        """Get non-filter parameters"""
        filter_param_ids = {id(p) for p in self.get_filter_parameters()}
        return [p for p in self.parameters() if id(p) not in filter_param_ids]
    
    def get_parameter_count(self):
        """Parameter count breakdown"""
        total = sum(p.numel() for p in self.parameters())
        filter_params = sum(p.numel() for p in self.get_filter_parameters())
        
        return {
            'total': total,
            'filter': filter_params,
            'combination': self.combination_weights.numel(),
            'other': total - filter_params
        }
    
    def debug_filter_learning(self):
        """Debug output"""
        print(f"\n=== SVD FILTER DEBUG ===")
        print(f"Filter Design: {self.filter}")
        print(f"User SVD Components: {self.u_n_components}")
        print(f"Item SVD Components: {self.i_n_components}")
        
        weights = torch.softmax(self.combination_weights, dim=0)
        print(f"Combination Weights: {weights.cpu().numpy()}")
        print("=== END SVD DEBUG ===\n")

    # For backward compatibility, keep eigenvalue terminology in public interface
    @property
    def u_n_eigen(self):
        """Backward compatibility: SVD components as eigenvalue count"""
        return self.u_n_components
    
    @property
    def i_n_eigen(self):
        """Backward compatibility: SVD components as eigenvalue count"""
        return self.i_n_components