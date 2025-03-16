# WanSeamlessFlow/blending.py

import torch
import math

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


def blend_embeddings(embed1, embed2, ratio, method="linear"):
    """
    Blend two embedding tensors using the specified interpolation method.
    
    Args:
        embed1: First embedding tensor
        embed2: Second embedding tensor
        ratio: Blend ratio (0.0-1.0)
        method: Interpolation method
        
    Returns:
        Blended embedding tensor
    """
    # Select interpolation function
    blend_func = getattr(BlendFunctions, method, BlendFunctions.linear)
    
    # Calculate interpolation factor
    factor = blend_func(ratio)
    
    # Perform embedding interpolation
    return embed1 * (1 - factor) + embed2 * factor