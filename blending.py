# WanSeamlessFlow/blending.py
import torch
import math
import comfy.model_management as mm
import numpy as np
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)

class BlendFunctions:
    """
    Embedding interpolation functions for smooth transitions.
    """

    @staticmethod
    def linear(ratio):
        """Linear interpolation - direct proportional blend"""
        return ratio

    @staticmethod
    def smooth(ratio):
        """Smoothstep function: 3x^2 - 2x^3"""
        return ratio * ratio * (3 - 2 * ratio)

    @staticmethod
    def ease_in(ratio):
        """Quadratic ease in - accelerating blend"""
        return ratio * ratio

    @staticmethod
    def ease_out(ratio):
        """Quadratic ease out - decelerating blend"""
        return ratio * (2 - ratio)

    @staticmethod
    def sine(ratio):
        """Sinusoidal easing - natural motion curve"""
        return 0.5 - 0.5 * math.cos(ratio * math.pi)

    @staticmethod
    def circ(ratio):
        """Circular easing - mimics circular motion"""
        return 1 - math.sqrt(1 - ratio * ratio)

    @staticmethod
    def bounce(ratio):
        """Bounce easing - mimics a bouncing effect"""
        n1 = 7.5625
        d1 = 2.75

        if ratio < 1 / d1:
            return n1 * ratio * ratio
        elif ratio < 2 / d1:
            ratio -= 1.5 / d1
            return n1 * ratio * ratio + 0.75
        elif ratio < 2.5 / d1:
            ratio -= 2.25 / d1
            return n1 * ratio * ratio + 0.9375
        else:
            ratio -= 2.625 / d1
            return n1 * ratio * ratio + 0.984375


def harmonize_embeddings(embeddings, target_length=None, strategy="pad_or_truncate"):
    """
    Make embeddings compatible for blending by normalizing their shapes.

    Args:
        embeddings: List of embedding tensors with potentially different shapes
        target_length: Target sequence length (if None, uses the maximum length)
        strategy: 'pad_or_truncate', 'mean_pooling', or 'weighted_tokens'

    Returns:
        List of compatible embeddings with identical shapes
    """
    if not embeddings or len(embeddings) < 2:
        return embeddings

    # Get device and dtype info
    device = embeddings[0].device
    dtype = embeddings[0].dtype

    # Identify the feature dimension (last dimension)
    feature_dim = embeddings[0].shape[-1]

    # Get sequence lengths
    seq_lengths = [e.shape[0] for e in embeddings]

    # Determine target length if not specified
    if target_length is None:
        if strategy == "pad_or_truncate":
            target_length = max(seq_lengths)
        else:
            target_length = min(seq_lengths)

    # Process each embedding based on strategy
    normalized = []
    for i, embed in enumerate(embeddings):
        curr_len = seq_lengths[i]

        if strategy == "pad_or_truncate":
            if curr_len < target_length:
                # Pad with zeros
                padding = torch.zeros(
                    target_length - curr_len, feature_dim, device=device, dtype=dtype
                )
                normalized.append(torch.cat([embed, padding], dim=0))
            elif curr_len > target_length:
                # Truncate to target length
                normalized.append(embed[:target_length])
            else:
                # Already the right length
                normalized.append(embed)

        elif strategy == "mean_pooling":
            # Mean pooling to unified representation
            mean_vector = torch.mean(embed, dim=0, keepdim=True)
            expanded = mean_vector.expand(target_length, -1)
            normalized.append(expanded)

        elif strategy == "weighted_tokens":
            # Create a weighted representation with importance based on position
            if curr_len < target_length:
                # Stretch fewer tokens to more
                indices = torch.linspace(0, curr_len - 1, target_length).long()
                normalized.append(embed[indices])
            elif curr_len > target_length:
                # Compress more tokens to fewer using weighted average
                indices = torch.linspace(0, curr_len - 1, target_length)
                floor_indices = indices.floor().long()
                ceil_indices = indices.ceil().long().clamp_max(curr_len - 1)
                fractions = indices - floor_indices.float()

                result = torch.zeros(
                    target_length, feature_dim, device=device, dtype=dtype
                )
                for j in range(target_length):
                    if floor_indices[j] == ceil_indices[j]:
                        result[j] = embed[floor_indices[j]]
                    else:
                        result[j] = (
                            embed[floor_indices[j]] * (1 - fractions[j])
                            + embed[ceil_indices[j]] * fractions[j]
                        )
                normalized.append(result)
            else:
                normalized.append(embed)

    return normalized


def blend_embeddings(embed1, embed2, ratio, method="linear"):
    """
    Blend two embedding tensors using the specified interpolation method.
    Automatically handles shape differences by normalizing first.

    Args:
        embed1: First embedding tensor
        embed2: Second embedding tensor
        ratio: Blend ratio (0.0-1.0)
        method: Interpolation method

    Returns:
        Blended embedding tensor
    """
    # Check for shape mismatch
    if embed1.shape[0] != embed2.shape[0]:
        # Normalize embeddings to make them compatible
        try:
            normalized = harmonize_embeddings([embed1, embed2])
            embed1, embed2 = normalized
        except Exception as e:
            log.info(f"Cannot blend embeddings due to shape mismatch: {str(e)}")
            # Fallback if we can't normalize
            return embed1

    # Select interpolation function
    blend_func = getattr(BlendFunctions, method, BlendFunctions.linear)

    # Calculate interpolation factor
    factor = blend_func(ratio)

    # Perform embedding interpolation
    return embed1 * (1 - factor) + embed2 * factor