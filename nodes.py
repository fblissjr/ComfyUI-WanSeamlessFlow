# WanSeamlessFlow/nodes.py

from .blending import blend_embeddings, harmonize_embeddings
import torch
import comfy.model_management as mm
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import gc

class WanSmartBlend:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text_embeds": ("WANVIDEOTEXTEMBEDS",),
                "blend_width": (
                    "INT",
                    {
                        "default": 8,
                        "min": 0,
                        "max": 32,
                        "step": 1,
                        "tooltip": "Width of transition zone in frames (latent space)",
                    },
                ),
                "blend_method": (
                    ["linear", "smooth", "ease_in", "ease_out", "sine", "circ"],
                    {
                        "default": "smooth",
                        "tooltip": "Interpolation curve between prompts",
                    },
                ),
                "normalization": (
                    ["pad_or_truncate", "mean_pooling", "weighted_tokens"],
                    {
                        "default": "pad_or_truncate",
                        "tooltip": "Method to harmonize different embedding shapes",
                    },
                ),
            },
            "optional": {
                "optimize_order": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Optimize embedding order to minimize semantic distance",
                    },
                ),
                "verbosity": (
                    "INT",
                    {
                        "default": 1,
                        "min": 0,
                        "max": 3,
                        "step": 1,
                        "tooltip": "0: None, 1: Basic, 2: Detailed, 3: Debug",
                    },
                ),
            },
        }

    RETURN_TYPES = ("WANVIDEOTEXTEMBEDS",)
    RETURN_NAMES = ("text_embeds",)
    FUNCTION = "process"
    CATEGORY = "WanSeamlessFlow"
    DESCRIPTION = "Prepares text embeddings for smooth transitions between context windows"

    def process(
        self,
        text_embeds: Dict[str, Any],
        blend_width: int,
        blend_method: str,
        normalization: str,
        optimize_order: bool = True,
        verbosity: int = 1,
    ) -> Tuple[Dict[str, Any]]:
        """
        Process text embeddings for optimal transitions.
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

        # Skip processing for single embedding
        if len(result["prompt_embeds"]) <= 1:
            if verbosity > 0:
                print("WanSmartBlend: Only one prompt, skipping processing")
            result["blend_width"] = blend_width
            result["blend_method"] = blend_method
            result["normalization"] = normalization
            result["verbosity"] = verbosity
            return (result,)

        # Check for shape inconsistencies
        shapes = [embed.shape for embed in result["prompt_embeds"]]
        if len(set(str(s) for s in shapes)) > 1:
            if verbosity > 0:
                print(f"WanSmartBlend: Detected varying embedding shapes: {shapes}")
                print(
                    f"WanSmartBlend: Normalizing embeddings for compatibility using '{normalization}' strategy..."
                )

            # Harmonize all embeddings to common shape
            try:
                result["prompt_embeds"] = harmonize_embeddings(
                    result["prompt_embeds"], strategy=normalization
                )

                if verbosity > 0:
                    new_shapes = [e.shape for e in result["prompt_embeds"]]
                    print(f"WanSmartBlend: Successfully normalized to {new_shapes[0]}")
            except Exception as e:
                print(f"WanSmartBlend: Error during harmonization: {str(e)}")
                # Reset to original if harmonization fails
                result["blend_width"] = 0  # Disable blending

        # Optimize embedding order if enabled
        if optimize_order and len(result["prompt_embeds"]) > 1:
            try:
                # Calculate embedding centroids, ensuring compatible dtype for numpy
                means = []
                for embed in result["prompt_embeds"]:
                    # Convert to float32 before numpy conversion to handle all tensor types
                    mean_embed = (
                        torch.mean(embed, dim=0).to(torch.float32).cpu().numpy()
                    )
                    means.append(mean_embed)

                # Nearest neighbor ordering
                order = [0]  # Start with first prompt
                remaining = set(range(1, len(means)))

                while remaining:
                    curr, best_dist = order[-1], float("inf")
                    best_next = None

                    for i in remaining:
                        # Calculate distance between current and candidate
                        dist = np.sum((means[curr] - means[i]) ** 2)
                        if dist < best_dist:
                            best_dist, best_next = dist, i

                    order.append(best_next)
                    remaining.remove(best_next)

                # Apply the optimized ordering
                result["prompt_embeds"] = [result["prompt_embeds"][i] for i in order]

                if verbosity > 0:
                    print(f"WanSmartBlend: Optimized prompt order: {order}")
            except Exception as e:
                print(f"WanSmartBlend: Error during optimization: {str(e)}")

        # Clean up memory
        gc.collect()
        mm.soft_empty_cache()
        
        # Add blend parameters to the result
        result["blend_width"] = blend_width
        result["blend_method"] = blend_method
        result["normalization"] = normalization
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