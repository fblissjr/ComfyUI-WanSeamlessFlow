# WanSeamlessFlow/utils/optimization.py

import numpy as np
import torch
from typing import List, Tuple

def calc_embedding_distance(embed1, embed2):
    """
    Calculate distance between two embeddings.
    
    Args:
        embed1: First embedding tensor
        embed2: Second embedding tensor
        
    Returns:
        Distance measure (lower means more similar)
    """
    # Move to CPU for numpy compatibility
    e1 = embed1.cpu().numpy()
    e2 = embed2.cpu().numpy()
    
    # Calculate L2 distance
    return np.sum((e1 - e2)**2)


def optimize_embedding_order(embeddings: List[torch.Tensor]) -> List[int]:
    """
    Optimize the order of embeddings to minimize semantic distance.
    Handles all dtype formats including BFloat16.
    
    Args:
        embeddings: List of embedding tensors
        
    Returns:
        Optimized ordering indices
    """
    if len(embeddings) <= 1:
        return list(range(len(embeddings)))
    
    # Calculate embedding centroids with proper dtype conversion
    means = []
    for embed in embeddings:
        # Convert to float32 before numpy conversion for compatibility
        mean_embed = torch.mean(embed, dim=0).to(torch.float32).cpu().numpy()
        means.append(mean_embed)
    
    # Nearest neighbor ordering
    order = [0]
    remaining = set(range(1, len(means)))
    
    while remaining:
        curr, best_dist = order[-1], float('inf')
        best_next = None
        
        for i in remaining:
            # Calculate distance between current and candidate
            dist = np.sum((means[curr] - means[i])**2)
            if dist < best_dist:
                best_dist, best_next = dist, i
        
        order.append(best_next)
        remaining.remove(best_next)
    
    return order


def compute_transition_cost_matrix(embeddings: List[torch.Tensor]) -> np.ndarray:
    """
    Compute full transition cost matrix between all embeddings.
    
    Args:
        embeddings: List of embedding tensors
        
    Returns:
        Matrix where cost_matrix[i][j] is transition cost from i to j
    """
    n = len(embeddings)
    cost_matrix = np.zeros((n, n), dtype=np.float32)
    
    # Calculate centroids for efficiency
    means = [torch.mean(embed, dim=0).to(torch.float32).cpu().numpy() for embed in embeddings]
    
    # Compute all pairwise distances
    for i in range(n):
        for j in range(n):
            if i != j:
                cost_matrix[i, j] = np.sum((means[i] - means[j])**2)
    
    return cost_matrix