# WanSeamlessFlow/nodes.py

import numpy as np
import torch
import comfy.model_management as mm
from typing import List, Dict, Any, Tuple, Optional

from .blending import blend_embeddings
from .utils.optimization import optimize_embedding_order, compute_transition_cost_matrix
from .visualization import create_transition_visualization

class WanSmartBlend:
    """
    Optimize text embeddings for seamless transitions between context windows.
    
    This node takes multiple text embeddings and:
    1. Optionally reorders them using nearest-neighbor optimization
    2. Adds blend parameters for smooth transitions
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text_embeds": ("WANVIDEOTEXTEMBEDS",),
                "blend_width": ("INT", {
                    "default": 8, "min": 0, "max": 32, "step": 1, 
                    "tooltip": "Width of transition zone in frames (latent space)"
                }),
                "blend_method": (
                    ["linear", "smooth", "ease_in", "ease_out", "sine", "circ", "bounce"], 
                    {
                        "default": "smooth",
                        "tooltip": "Interpolation curve between prompts"
                    }
                ),
            },
            "optional": {
                "optimize_order": ("BOOLEAN", {
                    "default": True, 
                    "tooltip": "Optimize embedding order to minimize semantic distance"
                }),
                "verbosity": ("INT", {
                    "default": 1, "min": 0, "max": 3, "step": 1,
                    "tooltip": "0: None, 1: Basic, 2: Detailed, 3: Debug"
                }),
            }
        }

    RETURN_TYPES = ("WANVIDEOTEXTEMBEDS",)
    RETURN_NAMES = ("text_embeds",)
    FUNCTION = "process"
    CATEGORY = "WanSeamlessFlow"
    DESCRIPTION = "Prepares text embeddings for smooth transitions between context windows"

    def process(self, 
                text_embeds: Dict[str, Any], 
                blend_width: int, 
                blend_method: str,
                optimize_order: bool = True,
                verbosity: int = 1) -> Tuple[Dict[str, Any]]:
        """
        Process text embeddings for optimal transitions.
        
        Args:
            text_embeds: Dictionary containing prompt embeddings
            blend_width: Width of transition zone in frames
            blend_method: Type of interpolation curve
            optimize_order: Whether to optimize embedding order
            verbosity: Level of debug information
            
        Returns:
            Enhanced text embeddings with blend parameters
        """
        # Create a deep copy to avoid modifying original
        result = {}
        for k, v in text_embeds.items():
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], torch.Tensor):
                result[k] = [t.clone() for t in v]
            elif isinstance(v, torch.Tensor):
                result[k] = v.clone()
            else:
                result[k] = v
        
        # Skip optimization for single embedding
        if len(result["prompt_embeds"]) <= 1 or not optimize_order:
            if verbosity > 0 and len(result["prompt_embeds"]) <= 1:
                print("WanSmartBlend: Only one prompt, skipping optimization")
            # Still add blend parameters for interface consistency
            result["blend_width"] = blend_width
            result["blend_method"] = blend_method
            result["verbosity"] = verbosity
            return (result,)
        
        # Log the operation
        if verbosity > 0:
            print(f"WanSmartBlend: Optimizing {len(result['prompt_embeds'])} embeddings with blend width {blend_width}")
        
        # Compute cost matrix for visualization
        if verbosity > 1:
            cost_matrix = compute_transition_cost_matrix(result["prompt_embeds"])
            print("\nTransition costs between prompts:")
            for i in range(len(cost_matrix)):
                print(f"  Prompt {i}: " + " ".join(f"{cost:.2f}" for cost in cost_matrix[i]))
        
        # Optimize embedding order
        if optimize_order and len(result["prompt_embeds"]) > 1:
            order = optimize_embedding_order(result["prompt_embeds"])
            
            # Apply the optimized ordering
            result["prompt_embeds"] = [result["prompt_embeds"][i] for i in order]
            
            # Log the final order
            if verbosity > 0:
                print(f"WanSmartBlend: Optimized prompt order: {order}")
        
        # Add blend parameters to the result
        result["blend_width"] = blend_width
        result["blend_method"] = blend_method
        result["verbosity"] = verbosity
        
        return (result,)


class WanBlendVisualize:
    """
    Diagnostic node to visualize the transition points between prompts.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text_embeds": ("WANVIDEOTEXTEMBEDS",),
                "num_frames": ("INT", {
                    "default": 81, "min": 1, "max": 1000, "step": 1,
                    "tooltip": "Total number of frames in the video"
                }),
                "show_blend_zones": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Highlight the blend zones"
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("blend_info",)
    FUNCTION = "process"
    CATEGORY = "WanSeamlessFlow"
    DESCRIPTION = "Generates a visualization of prompt transitions"

    def process(self, 
                text_embeds: Dict[str, Any], 
                num_frames: int,
                show_blend_zones: bool) -> Tuple[str]:
        """
        Visualize prompt transitions in a textual format.
        
        Args:
            text_embeds: Dictionary containing prompt embeddings
            num_frames: Total number of frames
            show_blend_zones: Whether to highlight blend zones
            
        Returns:
            Textual visualization of transitions
        """
        blend_width = text_embeds.get("blend_width", 0)
        blend_method = text_embeds.get("blend_method", "linear")
        num_prompts = len(text_embeds["prompt_embeds"])
        
        # Generate visualization
        visualization = create_transition_visualization(
            num_frames=num_frames,
            num_prompts=num_prompts,
            blend_width=blend_width,
            blend_method=blend_method,
            show_blend_zones=show_blend_zones
        )
        
        return (visualization,)


# Node mapping for ComfyUI
NODE_CLASS_MAPPINGS = {
    "WanSmartBlend": WanSmartBlend,
    "WanBlendVisualize": WanBlendVisualize,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanSmartBlend": "Wan Smart Blend",
    "WanBlendVisualize": "Wan Blend Visualize",
}