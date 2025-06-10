"""
Enhanced uSpec with Multi-View Adjacency Normalization - SVD Optimized
Combines PolyCF's multi-view approach with uSpec's advanced spectral filtering
Now using computationally efficient SVD instead of eigendecomposition
"""

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


class MultiViewUniversalSpectralCF(nn.Module):
    """Enhanced Universal Spectral CF with Multi-View Adjacency Normalization - SVD Optimized"""
    
    def __init__(self, adj_mat, config=None):
        super().__init__()
        
        print("🚀 Initializing Multi-View Universal Spectral CF with SVD optimization...")
        
        self.config = config or {}
        self.device = self.config.get('device', 'cpu')
        self.filter_order = self.config.get('filter_order', 6)
        
        # Multi-view configuration
        self.gamma_values = self.config.get('gamma_values', [0.0, 0.5, 1.0])
        self.use_similarity_views = self.config.get('use_similarity_views', True)
        self.use_adjacency_views = self.config.get('use_adjacency_views', True)
        
        print(f"📊 Multi-view configuration:")
        print(f"   ├─ Adjacency views (PolyCF): {self.use_adjacency_views}")
        print(f"   ├─ Similarity views (uSpec): {self.use_similarity_views}")
        if self.use_adjacency_views:
            print(f"   ├─ Gamma values: {self.gamma_values}")
        print(f"   └─ Total views: {self._count_total_views()}")
        
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
        
        # Setup multi-view components
        print("\n🔧 Setting up multi-view spectral filters with SVD...")
        self._setup_multiview_filters()
        
        print("🎯 Initializing view combination weights...")
        self._setup_view_combination_weights()
        
        print("✅ Multi-view model initialization complete!\n")
    
    def _count_total_views(self):
        """Count total number of views"""
        view_count = 0
        if self.use_adjacency_views:
            view_count += len(self.gamma_values) * 2  # User and item gram matrices for each gamma
        if self.use_similarity_views:
            view_count += 2  # User and item similarity matrices
        return view_count
    
    def _create_polycf_adjacency_views(self):
        """Create PolyCF-style adjacency views with different gamma normalizations"""
        print("🎭 Creating PolyCF adjacency views...")
        
        adjacency_views = {}
        adj_coo = sp.coo_matrix(self.adj_tensor.cpu().numpy())
        
        # Compute degree matrices
        user_degrees = np.array(adj_coo.sum(axis=1)).flatten()
        item_degrees = np.array(adj_coo.sum(axis=0)).flatten()
        
        # Add small epsilon to avoid division by zero
        user_degrees = user_degrees + 1e-8
        item_degrees = item_degrees + 1e-8
        
        for gamma in self.gamma_values:
            print(f"   ├─ Creating view with γ={gamma:.1f}")
            
            # PolyCF normalization: R_γ = D_r^(-γ) R D_c^(-(1-γ))
            user_norm_power = -gamma
            item_norm_power = -(1 - gamma)
            
            # Create normalization matrices
            user_norm = sp.diags(np.power(user_degrees, user_norm_power))
            item_norm = sp.diags(np.power(item_degrees, item_norm_power))
            
            # Apply normalization: D_r^(-γ) R D_c^(-(1-γ))
            normalized_adj = user_norm @ adj_coo @ item_norm
            
            # Convert to dense tensor for SVD
            normalized_dense = torch.tensor(normalized_adj.toarray(), dtype=torch.float32, device=self.device)
            
            # Create gram matrices for this view
            # User gram: R_γ @ R_γ.T (projected to item space: R_γ.T @ R_γ @ R_γ.T @ R_γ)
            user_gram = torch.mm(normalized_dense, normalized_dense.t())
            user_gram_proj = torch.mm(torch.mm(normalized_dense.t(), user_gram), normalized_dense)
            
            # Item gram: R_γ.T @ R_γ
            item_gram = torch.mm(normalized_dense.t(), normalized_dense)
            
            adjacency_views[f'user_gram_gamma_{gamma}'] = user_gram_proj
            adjacency_views[f'item_gram_gamma_{gamma}'] = item_gram
            
            print(f"   │  ├─ User gram projection: {user_gram_proj.shape}")
            print(f"   │  └─ Item gram: {item_gram.shape}")
        
        print(f"   └─ Created {len(adjacency_views)} adjacency view matrices")
        return adjacency_views
    
    def _create_similarity_views(self):
        """Create similarity-based views (original uSpec approach)"""
        print("🔗 Creating similarity-based views...")
        
        similarity_views = {}
        
        # User-user similarity
        print("   ├─ Computing user-user similarity...")
        user_similarity = self._compute_similarity_matrix(self.adj_tensor, 'user')
        
        # Check for valid similarity data
        has_data = False
        if hasattr(user_similarity, 'nnz'):  # Sparse tensor
            has_data = user_similarity.nnz > 0
        elif hasattr(user_similarity, 'numel'):  # Dense tensor
            has_data = user_similarity.numel() > 0 and torch.sum(user_similarity) > 0
        else:  # Fallback
            has_data = True
            
        if has_data:
            user_sim_laplacian = self._compute_similarity_laplacian(user_similarity)
            similarity_views['user_similarity'] = user_sim_laplacian
            print(f"   │  └─ User similarity matrix: {user_sim_laplacian.shape}")
        else:
            print(f"   │  └─ User similarity matrix: empty, skipping")
        
        # Item-item similarity  
        print("   ├─ Computing item-item similarity...")
        item_similarity = self._compute_similarity_matrix(self.adj_tensor.t(), 'item')
        
        # Check for valid similarity data
        has_data = False
        if hasattr(item_similarity, 'nnz'):  # Sparse tensor
            has_data = item_similarity.nnz > 0
        elif hasattr(item_similarity, 'numel'):  # Dense tensor
            has_data = item_similarity.numel() > 0 and torch.sum(item_similarity) > 0
        else:  # Fallback
            has_data = True
            
        if has_data:
            item_sim_laplacian = self._compute_similarity_laplacian(item_similarity)
            similarity_views['item_similarity'] = item_sim_laplacian
            print(f"   │  └─ Item similarity matrix: {item_sim_laplacian.shape}")
        else:
            print(f"   │  └─ Item similarity matrix: empty, skipping")
        
        print(f"   └─ Created {len(similarity_views)} similarity view matrices")
        return similarity_views
    
    def _compute_similarity_matrix(self, interaction_matrix, cache_type=None):
        """Compute similarity matrix with progress and caching"""
        print(f"    🔍 Computing {cache_type} similarity matrix...")
        
        # Check cache first
        cache_path = self._get_cache_path('sim', cache_type) if cache_type else None
        if cache_path and os.path.exists(cache_path):
            try:
                print(f"    📂 Loading from cache: {os.path.basename(cache_path)}")
                with open(cache_path, 'rb') as f:
                    similarity = pickle.load(f)
                print(f"    ✅ Cache loaded successfully")
                return similarity.to(self.device)
            except Exception as e:
                print(f"    ⚠️  Cache loading failed: {e}")
        
        # Compute A @ A.T
        with torch.no_grad():
            similarity = torch.mm(interaction_matrix, interaction_matrix.t())
            
            # Apply threshold
            if self.similarity_threshold > 0:
                similarity = torch.where(similarity >= self.similarity_threshold, 
                                       similarity, torch.zeros_like(similarity))
            
            # Set diagonal and clamp
            similarity.fill_diagonal_(1.0)
            similarity = torch.clamp(similarity, min=0.0, max=1.0)
        
        # Save to cache
        if cache_path:
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(similarity.cpu(), f)
            except Exception as e:
                print(f"    ⚠️  Cache saving failed: {e}")
        
        return similarity
    
    def _compute_similarity_laplacian(self, similarity_matrix):
        """Compute Laplacian from similarity matrix"""
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
        
        return laplacian
    
    def _get_cache_path(self, cache_type, filter_type=None):
        """Generate cache path"""
        cache_dir = "../cache"
        os.makedirs(cache_dir, exist_ok=True)
        
        dataset = self.config.get('dataset', 'unknown')
        threshold = str(self.similarity_threshold).replace('.', 'p')
        
        if filter_type:
            k_value = self.u_n_components if filter_type == 'user' else self.i_n_components
            filename = f"{dataset}_mv_svd_{filter_type}_{cache_type}_k{k_value}_th{threshold}.pkl"
        else:
            filename = f"{dataset}_mv_svd_{cache_type}_th{threshold}.pkl"
            
        return os.path.join(cache_dir, filename)
    
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
        elif sparsity > 0.05:
            multiplier = 0.8
        else:
            multiplier = 1.0
            
        u_components = int(base_u * multiplier)
        i_components = int(base_i * multiplier)
        
        print(f"   ├─ User SVD components: {base_u} → {u_components}")
        print(f"   └─ Item SVD components: {base_i} → {i_components}")
        
        return u_components, i_components
    
    def _setup_multiview_filters(self):
        """Setup filters for all views with detailed progress"""
        all_views = {}
        
        # Add adjacency views (PolyCF style)
        if self.use_adjacency_views:
            adjacency_views = self._create_polycf_adjacency_views()
            all_views.update(adjacency_views)
        
        # Add similarity views (original uSpec style)
        if self.use_similarity_views:
            similarity_views = self._create_similarity_views()
            all_views.update(similarity_views)
        
        print(f"\n🎭 Setting up SVD filters for {len(all_views)} views:")
        for view_name in all_views.keys():
            print(f"   ├─ {view_name}")
        
        # Setup SVD decompositions and filters for all views
        self.view_svd_data = {}
        self.view_filters = {}
        
        for view_name, matrix in all_views.items():
            print(f"\n🔧 Processing view: {view_name}")
            
            # Determine matrix type for component count
            is_user_matrix = 'user' in view_name
            k = self.u_n_components if is_user_matrix else self.i_n_components
            
            # Compute SVD decomposition
            singular_vals, svd_vectors = self._compute_svd_cached(matrix, k, view_name)
            
            self.view_svd_data[view_name] = {
                'singular_vals': torch.tensor(singular_vals, dtype=torch.float32, device=self.device),
                'svd_vectors': torch.tensor(svd_vectors, dtype=torch.float32, device=self.device)
            }
            
            # Create filter for this view
            self.view_filters[view_name] = self._create_filter_instance()
            print(f"   ├─ {view_name}: {len(singular_vals)} components")
        
        print(f"\n✅ All {len(all_views)} view filters setup complete")
    
    def _compute_svd_cached(self, matrix, k, view_name):
        """Compute SVD decomposition with caching - much faster than eigendecomposition"""
        cache_path = self._get_cache_path('svd', view_name)
        
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    singular_vals, svd_vectors = pickle.load(f)
                print(f"   📂 Loaded SVD decomposition from cache")
                return singular_vals, svd_vectors
            except Exception as e:
                print(f"   ⚠️  Cache loading failed: {e}")
        
        print(f"   ⚡ Computing SVD decomposition (much faster than eigendecomposition)...")
        
        try:
            # Convert to numpy for SVD
            if torch.is_tensor(matrix):
                matrix_np = matrix.cpu().numpy()
            else:
                matrix_np = matrix.toarray() if sp.issparse(matrix) else matrix
            
            print(f"   📊 Matrix info: shape={matrix_np.shape}, dtype={matrix_np.dtype}")
            print(f"   📊 Matrix stats: min={matrix_np.min():.6f}, max={matrix_np.max():.6f}")
            
            # Check for problematic matrices
            if not np.all(np.isfinite(matrix_np)):
                print(f"   ❌ Matrix contains inf/nan values, using fallback")
                return self._fallback_svd_decomposition(matrix_np.shape[0], k)
            
            if np.allclose(matrix_np, 0):
                print(f"   ❌ Matrix is all zeros, using fallback")
                return self._fallback_svd_decomposition(matrix_np.shape[0], k)
            
            # Ensure matrix is symmetric and positive semidefinite for spectral filtering
            matrix_np = (matrix_np + matrix_np.T) / 2  # Force symmetry
            
            # Add small regularization to diagonal for numerical stability
            np.fill_diagonal(matrix_np, matrix_np.diagonal() + 1e-6)
            
            # Reduce k for safety and efficiency
            k = min(k, matrix_np.shape[0] - 2, 50)  # Cap at 50 components max
            print(f"   🎯 Using k={k} SVD components for efficiency")
            
            print(f"   ⏳ Starting SVD decomposition...")
            start_time = time.time()
            
            # Use TruncatedSVD for efficiency (much faster than full SVD)
            if k < matrix_np.shape[0] - 1:
                try:
                    # TruncatedSVD is much faster for k << n
                    svd_solver = TruncatedSVD(n_components=k, random_state=42)
                    
                    # Fit the SVD solver
                    svd_solver.fit(matrix_np)
                    
                    # Extract singular values and vectors
                    singular_vals = svd_solver.singular_values_
                    svd_vectors = svd_solver.components_.T  # Transpose to get right singular vectors
                    
                    # Convert singular values to "eigenvalue-like" values for spectral filtering
                    # For symmetric matrices: eigenvalues ≈ singular_values
                    singular_vals = np.maximum(singular_vals, 0.0)
                    
                    svd_time = time.time() - start_time
                    print(f"   ⚡ TruncatedSVD completed in {svd_time:.2f}s (much faster!)")
                    
                except Exception as svd_error:
                    print(f"   ⚠️  TruncatedSVD failed: {svd_error}, trying scipy SVD...")
                    
                    # Fallback to scipy SVD
                    U, s, Vt = scipy_svd(matrix_np, full_matrices=False)
                    singular_vals = s[:k]
                    svd_vectors = U[:, :k]
                    
                    svd_time = time.time() - start_time
                    print(f"   ✅ Scipy SVD completed in {svd_time:.2f}s")
            else:
                # For very small matrices, use full SVD
                U, s, Vt = scipy_svd(matrix_np, full_matrices=False)
                singular_vals = s[:k]
                svd_vectors = U[:, :k]
                
                svd_time = time.time() - start_time
                print(f"   ✅ Full SVD completed in {svd_time:.2f}s")
            
            print(f"   ✅ Success! {k} components in range [{singular_vals.min():.4f}, {singular_vals.max():.4f}]")
            
            # Cache results
            try:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                with open(cache_path, 'wb') as f:
                    pickle.dump((singular_vals, svd_vectors), f)
                print(f"   💾 Cached SVD decomposition")
            except Exception as e:
                print(f"   ⚠️  Cache saving failed: {e}")
            
        except Exception as e:
            print(f"   ❌ SVD decomposition failed: {e}")
            print(f"   🔄 Using fallback method...")
            return self._fallback_svd_decomposition(matrix_np.shape[0], k)
        
        return singular_vals, svd_vectors
    
    def _fallback_svd_decomposition(self, matrix_size, k):
        """Fallback when SVD fails"""
        print(f"   🔄 Generating {k} fallback SVD components...")
        k = min(k, 8)  # Very conservative
        singular_vals = np.linspace(0.01, 1.0, k)  # Avoid zeros
        svd_vectors = np.random.randn(matrix_size, k)
        svd_vectors, _ = np.linalg.qr(svd_vectors)
        return singular_vals, svd_vectors
    
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
    
    def _setup_view_combination_weights(self):
        """Setup combination weights for all views"""
        n_views = len(self.view_filters)
        
        if n_views > 1:
            # Initialize with equal weights, but add some structure
            weights = torch.ones(n_views)
            
            # Give slightly higher weight to adjacency views if both types are present
            if self.use_adjacency_views and self.use_similarity_views:
                adj_view_count = len(self.gamma_values) * 2
                sim_view_count = 2
                
                # Slightly favor adjacency views (PolyCF approach)
                for i, view_name in enumerate(self.view_filters.keys()):
                    if 'gamma' in view_name:
                        weights[i] = 1.2  # Slight boost for PolyCF views
            
            weights = weights / weights.sum()  # Normalize
            print(f"  🎚️  Initialized {n_views} view weights: {weights.tolist()}")
        else:
            weights = torch.ones(1)
            print(f"  🎚️  Single view weight: [1.0]")
        
        self.view_combination_weights = nn.Parameter(weights.to(self.device))
    
    def _get_view_filter_matrices(self):
        """Compute filter matrices for all views using SVD data"""
        view_matrices = {}
        
        for view_name, svd_data in self.view_svd_data.items():
            singular_vals = svd_data['singular_vals']
            svd_vectors = svd_data['svd_vectors']
            
            # Apply spectral filter to singular values (treating them as eigenvalues)
            filter_response = self.view_filters[view_name](singular_vals)
            
            # Reconstruct filtered matrix: U @ diag(filtered_singular_vals) @ U.T
            filter_matrix = torch.mm(torch.mm(svd_vectors, torch.diag(filter_response)), 
                                   svd_vectors.t())
            view_matrices[view_name] = filter_matrix
        
        return view_matrices
    
    def forward(self, users):
        """Enhanced forward pass with multi-view combination"""
        if users.device != self.adj_tensor.device:
            users = users.to(self.adj_tensor.device)
        
        user_profiles = self.adj_tensor[users]
        view_filter_matrices = self._get_view_filter_matrices()
        
        view_scores = []
        
        for view_name, filter_matrix in view_filter_matrices.items():
            if 'user' in view_name and 'similarity' in view_name:
                # User similarity filtering: filter users then project to items
                batch_size = len(users)
                user_embeddings = torch.zeros(batch_size, self.n_users, device=self.adj_tensor.device)
                user_embeddings[range(batch_size), users] = 1.0
                
                filtered_users = user_embeddings @ filter_matrix
                scores = filtered_users @ self.adj_tensor
            
            elif 'item' in view_name:
                # Item filtering: filter items directly
                scores = user_profiles @ filter_matrix
            
            else:
                # Default: apply filter to user profiles
                scores = user_profiles @ filter_matrix
            
            view_scores.append(scores)
        
        # Combine all views with learned weights
        if len(view_scores) > 1:
            weights = torch.softmax(self.view_combination_weights, dim=0)
            predicted = sum(w * score for w, score in zip(weights, view_scores))
        else:
            predicted = view_scores[0]
        
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
    
    def get_view_parameters(self):
        """Get all view filter parameters"""
        params = []
        for filter_module in self.view_filters.values():
            params.extend(list(filter_module.parameters()))
        params.append(self.view_combination_weights)
        return params
    
    def get_parameter_count(self):
        """Parameter count breakdown"""
        total = sum(p.numel() for p in self.parameters())
        view_params = sum(p.numel() for p in self.get_view_parameters())
        
        return {
            'total': total,
            'view_filters': view_params - self.view_combination_weights.numel(),
            'view_combination': self.view_combination_weights.numel(),
            'other': total - view_params
        }
    
    def debug_view_learning(self):
        """Debug output for multi-view learning"""
        print(f"\n=== MULTI-VIEW SVD DEBUG ===")
        print(f"Filter Design: {self.filter}")
        print(f"Adjacency Views: {self.use_adjacency_views}")
        print(f"Similarity Views: {self.use_similarity_views}")
        print(f"Gamma Values: {self.gamma_values}")
        print(f"Total Views: {len(self.view_filters)}")
        print(f"SVD Components: u={self.u_n_components}, i={self.i_n_components}")
        
        weights = torch.softmax(self.view_combination_weights, dim=0)
        print(f"View Combination Weights:")
        for i, (view_name, weight) in enumerate(zip(self.view_filters.keys(), weights.detach().cpu().numpy())):
            print(f"  {view_name}: {weight:.4f}")
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