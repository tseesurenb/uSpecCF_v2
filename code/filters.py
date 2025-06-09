'''
Optimized Filter Designs
Removed unnecessary computations and complexity
'''

import torch
import torch.nn as nn
import numpy as np

# Simplified filter patterns
filter_patterns = {
    'smooth': [1.0, -0.5, 0.1, -0.02, 0.004, -0.0008, 0.00015],
    'butterworth': [1.0, -0.6, 0.2, -0.05, 0.01, -0.002, 0.0003],
    'gaussian': [1.0, -0.7, 0.15, -0.03, 0.005, -0.0007, 0.00008],
    'golden_036': [1.0, -0.36, 0.1296, -0.220, 0.1564, -0.088, 0.0548],
}

def get_filter_coefficients(filter_name, order=None, as_tensor=False):
    """Get filter coefficients"""
    if filter_name not in filter_patterns:
        filter_name = 'smooth'  # Default fallback
    
    coeffs = filter_patterns[filter_name].copy()
    
    if order is not None:
        if len(coeffs) > order + 1:
            coeffs = coeffs[:order + 1]
        elif len(coeffs) < order + 1:
            coeffs.extend([0.0] * (order + 1 - len(coeffs)))
    
    if as_tensor:
        return torch.tensor(coeffs, dtype=torch.float32)
    
    return coeffs

class UniversalSpectralFilter(nn.Module):
    """Optimized Universal Spectral Filter"""
    
    def __init__(self, filter_order=6, init_filter_name='smooth'):
        super().__init__()
        self.filter_order = filter_order
        
        # Initialize coefficients
        init_coeffs = get_filter_coefficients(init_filter_name, order=filter_order, as_tensor=True)
        self.coeffs = nn.Parameter(init_coeffs)
    
    def forward(self, eigenvalues):
        """Optimized Chebyshev polynomial evaluation"""
        # Normalize eigenvalues
        max_eigenval = torch.max(eigenvalues) + 1e-8
        x = 2 * (eigenvalues / max_eigenval) - 1
        
        # Efficient Chebyshev evaluation
        result = self.coeffs[0] * torch.ones_like(x)
        
        if len(self.coeffs) > 1:
            T_prev = torch.ones_like(x)
            T_curr = x
            result += self.coeffs[1] * T_curr
            
            for i in range(2, len(self.coeffs)):
                T_next = 2 * x * T_curr - T_prev
                result += self.coeffs[i] * T_next
                T_prev, T_curr = T_curr, T_next
        
        # Simplified response function
        return torch.sigmoid(result) + 1e-6

class EnhancedSpectralBasisFilter(nn.Module):
    """Optimized Enhanced Basis Filter"""
    
    def __init__(self, filter_order=6, init_filter_name='smooth'):
        super().__init__()
        self.filter_order = filter_order
        
        # Reduced filter bank
        filter_names = ['golden_036', 'smooth', 'butterworth']
        
        # Store filter coefficients as buffers
        self.filter_bank = []
        for i, name in enumerate(filter_names):
            coeffs = get_filter_coefficients(name, order=filter_order, as_tensor=True)
            if len(coeffs) < filter_order + 1:
                padded_coeffs = torch.zeros(filter_order + 1)
                padded_coeffs[:len(coeffs)] = coeffs
                coeffs = padded_coeffs
            elif len(coeffs) > filter_order + 1:
                coeffs = coeffs[:filter_order + 1]
            
            self.register_buffer(f'filter_{i}', coeffs)
            self.filter_bank.append(getattr(self, f'filter_{i}'))
        
        # Learnable parameters
        self.mixing_weights = nn.Parameter(torch.ones(len(filter_names)) / len(filter_names))
        self.refinement_coeffs = nn.Parameter(torch.zeros(filter_order + 1) * 0.1)
        self.refinement_scale = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, eigenvalues):
        """Optimized forward pass"""
        weights = torch.softmax(self.mixing_weights, dim=0)
        
        # Mix base filters
        mixed_coeffs = torch.zeros_like(self.filter_bank[0])
        for i, base_filter in enumerate(self.filter_bank):
            mixed_coeffs += weights[i] * base_filter
        
        # Add refinement
        final_coeffs = mixed_coeffs + self.refinement_scale * self.refinement_coeffs
        
        # Evaluate polynomial
        max_eigenval = torch.max(eigenvalues) + 1e-8
        x = 2 * (eigenvalues / max_eigenval) - 1
        
        result = final_coeffs[0] * torch.ones_like(x)
        if len(final_coeffs) > 1:
            T_prev, T_curr = torch.ones_like(x), x
            result += final_coeffs[1] * T_curr
            
            for i in range(2, len(final_coeffs)):
                T_next = 2 * x * T_curr - T_prev
                result += final_coeffs[i] * T_next
                T_prev, T_curr = T_curr, T_next
        
        return torch.sigmoid(result) + 1e-6

class MultiScaleSpectralFilter(nn.Module):
    """Optimized Multi-scale Filter"""
    
    def __init__(self, filter_order=6, init_filter_name='smooth', n_bands=3):
        super().__init__()
        self.n_bands = n_bands
        
        # Learnable band parameters
        self.band_boundaries = nn.Parameter(torch.linspace(0.2, 0.8, n_bands - 1))
        self.band_responses = nn.Parameter(torch.ones(n_bands) * 0.5)
        self.transition_sharpness = nn.Parameter(torch.tensor(5.0))
    
    def forward(self, eigenvalues):
        """Optimized multi-band filtering"""
        max_eigenval = torch.max(eigenvalues) + 1e-8
        norm_eigenvals = eigenvalues / max_eigenval
        
        # Create band boundaries
        boundaries = torch.cat([
            torch.zeros(1, device=eigenvalues.device),
            torch.sort(self.band_boundaries)[0],
            torch.ones(1, device=eigenvalues.device)
        ])
        
        sharpness = torch.abs(self.transition_sharpness) + 1.0
        band_responses = torch.sigmoid(self.band_responses)
        
        response = torch.zeros_like(norm_eigenvals)
        
        # Efficient band assignment
        for i in range(self.n_bands):
            left_boundary = boundaries[i]
            right_boundary = boundaries[i + 1]
            
            # Smooth band membership
            left_weight = torch.sigmoid(sharpness * (norm_eigenvals - left_boundary))
            right_weight = torch.sigmoid(sharpness * (right_boundary - norm_eigenvals))
            
            band_membership = left_weight * right_weight
            response += band_membership * band_responses[i]
        
        return torch.clamp(response, min=1e-6, max=1.0)

class EnsembleSpectralFilter(nn.Module):
    """Optimized Ensemble Filter"""
    
    def __init__(self, filter_order=6, init_filter_name='smooth'):
        super().__init__()
        
        # Reduced ensemble with most effective filters
        self.filter1 = UniversalSpectralFilter(filter_order, init_filter_name)
        self.filter2 = EnhancedSpectralBasisFilter(filter_order, init_filter_name)
        self.filter3 = MultiScaleSpectralFilter(filter_order, init_filter_name)
        
        # Ensemble weights
        self.ensemble_weights = nn.Parameter(torch.ones(3) / 3)
    
    def forward(self, eigenvalues):
        """Optimized ensemble forward"""
        response1 = self.filter1(eigenvalues)
        response2 = self.filter2(eigenvalues)
        response3 = self.filter3(eigenvalues)
        
        weights = torch.softmax(self.ensemble_weights, dim=0)
        
        return (weights[0] * response1 + 
                weights[1] * response2 + 
                weights[2] * response3)

# Export optimized filters
__all__ = [
    'UniversalSpectralFilter',
    'EnhancedSpectralBasisFilter', 
    'MultiScaleSpectralFilter',
    'EnsembleSpectralFilter',
    'get_filter_coefficients',
    'filter_patterns'
]