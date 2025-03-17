# WanSeamlessFlow/nodes.py

from .blending import blend_embeddings, harmonize_embeddings
import torch
import comfy.model_management as mm
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import gc
import logging
import numpy as np
import torch
import gc
from typing import List, Dict, Any, Tuple, Optional
import comfy.model_management as mm
import traceback

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)


class WanSmartBlend:
    """
    Normalize text embeddings and prepare them for smooth transitions between context windows.
    """

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
                    ["linear", "smooth", "ease_in", "ease_out"],
                    {
                        "default": "smooth",
                        "tooltip": "Interpolation curve between prompts",
                    },
                ),
                "normalization": (
                    ["pad_truncate", "none"],
                    {
                        "default": "pad_truncate",
                        "tooltip": "How to handle different embedding shapes",
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

        Args:
            text_embeds: Dictionary containing prompt embeddings
            blend_width: Width of transition zone in frames
            blend_method: Type of interpolation curve
            normalize_strategy: How to handle shape differences
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

        # Skip optimization processing for single embedding
        if len(result["prompt_embeds"]) <= 1:
            if verbosity > 0:
                log.info(
                    "WanSmartBlend: Only one prompt, skipping optimization processing"
                )
            # Still add blend parameters for interface consistency
            result["blend_width"] = blend_width
            result["blend_method"] = blend_method
            result["normalization"] = normalization
            result["verbosity"] = verbosity
            return (result,)

        # Log the operation
        if verbosity > 0:
            log.info(
                f"WanSmartBlend: Optimizing {len(result['prompt_embeds'])} embeddings with blend width {blend_width}"
            )

        # Check for shape inconsistencies
        shapes = [embed.shape for embed in result["prompt_embeds"]]
        if len(set(str(s) for s in shapes)) > 1:
            if verbosity > 0:
                log.info(f"WanSmartBlend: Detected varying embedding shapes: {shapes}")
                log.info(
                    f"WanSmartBlend: Normalizing embeddings for compatibility using '{normalization}' strategy..."
                )

            # Harmonize all embeddings to common shape
            try:
                # result["prompt_embeds"] = harmonize_embeddings(
                #     result["prompt_embeds"], strategy=normalization
                # )

                if normalization == "pad_truncate":
                    fixed_normalization = "pad_or_truncate"
                else:
                    fixed_normalization = normalization

                result["prompt_embeds"] = harmonize_embeddings(
                    result["prompt_embeds"], strategy=fixed_normalization
                )

                if verbosity > 0:
                    new_shapes = [e.shape for e in result["prompt_embeds"]]
                    log.info(
                        f"WanSmartBlend: Successfully normalized to {new_shapes[0]}"
                    )
            except Exception as e:
                log.info(f"WanSmartBlend: Error during harmonization: {str(e)}")
                # Reset to original if harmonization fails
                result["blend_width"] = 0  # Disable blending

            # Add after calling harmonize_embeddings:
            log.info(
                f"DEBUG: Before normalization - shapes: {[e.shape for e in result['prompt_embeds']]}"
            )
            try:
                # result["prompt_embeds"] = harmonize_embeddings(
                #     result["prompt_embeds"], strategy=normalization
                # )
                if normalization == "pad_truncate":
                    fixed_normalization = "pad_or_truncate"
                else:
                    fixed_normalization = normalization

                result["prompt_embeds"] = harmonize_embeddings(
                    result["prompt_embeds"], strategy=fixed_normalization
                )
                log.info(
                    f"DEBUG: After normalization - shapes: {[e.shape for e in result['prompt_embeds']]}"
                )
            except Exception as e:
                log.info(
                    f"DEBUG: Harmonization error: {str(e)}\n{traceback.format_exc()}"
                )
                result["blend_width"] = 0

        # After the harmonization check for issues
        if not result["prompt_embeds"] or len(result["prompt_embeds"]) == 0:
            log.info(
                "WARNING: Harmonization resulted in empty prompt array, restoring original"
            )
            # result["prompt_embeds"] = original_prompts.copy()  # Keep a copy of the original prompts
            result["blend_width"] = 0  # Disable blending since we can't normalize

        # Check embedding shape consistency
        shapes = [embed.shape for embed in result["prompt_embeds"]]
        if len(set(str(s) for s in shapes)) > 1:
            if verbosity > 0:
                log.info(f"WanSmartBlend: Detected varying embedding shapes: {shapes}")

            # Normalize embeddings if needed
            if normalization == "pad_truncate":
                if verbosity > 0:
                    log.info(
                        f"WanSmartBlend: Normalizing embeddings for compatibility..."
                    )

                # Find the target shape (max sequence length)
                max_seq_len = max(s[0] for s in shapes)
                feature_dim = shapes[0][1]  # Feature dimension is consistent

                # Normalize each embedding
                normalized_embeds = []
                for i, embed in enumerate(result["prompt_embeds"]):
                    curr_len = embed.shape[0]
                    if curr_len < max_seq_len:
                        # Pad with zeros
                        padding = torch.zeros(
                            max_seq_len - curr_len,
                            feature_dim,
                            device=embed.device,
                            dtype=embed.dtype,
                        )
                        normalized_embeds.append(torch.cat([embed, padding], dim=0))
                    elif curr_len > max_seq_len:
                        # Truncate
                        normalized_embeds.append(embed[:max_seq_len])
                    else:
                        # Already correct size
                        normalized_embeds.append(embed)

                result["prompt_embeds"] = normalized_embeds

                if verbosity > 0:
                    log.info(
                        f"WanSmartBlend: Successfully normalized all embeddings to shape {result['prompt_embeds'][0].shape}"
                    )

        # Calculate embedding centroids for optimization
        if optimize_order and len(result["prompt_embeds"]) > 1:
            # Calculate means with proper dtype handling
            means = []
            for embed in result["prompt_embeds"]:
                mean_embed = torch.mean(embed, dim=0).to(torch.float32).cpu().numpy()
                means.append(mean_embed)

            # Nearest neighbor ordering
            order = [0]  # Start with first prompt
            remaining = set(range(1, len(means)))

            while remaining:
                curr, best_dist = order[-1], float("inf")
                best_next = None

                for i in remaining:
                    # Calculate distance
                    dist = np.sum((means[curr] - means[i]) ** 2)
                    if dist < best_dist:
                        best_dist, best_next = dist, i

                order.append(best_next)
                remaining.remove(best_next)

            # Apply the optimized ordering
            result["prompt_embeds"] = [result["prompt_embeds"][i] for i in order]

            # Log the final order
            if verbosity > 0:
                log.info(f"WanSmartBlend: Optimized prompt order: {order}")
        
        # Add blend parameters to the result
        result["blend_width"] = blend_width
        result["blend_method"] = blend_method
        result["normalization"] = normalization
        result["verbosity"] = verbosity

        # Memory cleanup
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

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
                "num_frames": (
                    "INT",
                    {
                        "default": 81,
                        "min": 1,
                        "max": 1000,
                        "step": 1,
                        "tooltip": "Total number of frames in the video",
                    },
                ),
                "show_blend_zones": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Highlight the blend zones"},
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("blend_info",)
    FUNCTION = "process"
    CATEGORY = "WanSeamlessFlow"
    DESCRIPTION = "Generates a visualization of prompt transitions"

    def process(
        self, text_embeds: Dict[str, Any], num_frames: int, show_blend_zones: bool
    ) -> Tuple[str]:
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

        # Nothing to visualize if no blending or single prompt
        if blend_width == 0 or num_prompts <= 1:
            return ("No blend zones configured or only one prompt available.",)

        # Calculate section size
        section_size = num_frames / num_prompts

        # Build visualization
        visualization = []
        visualization.append(
            f"Transition visualization for {num_frames} frames with {num_prompts} prompts"
        )
        visualization.append(
            f"Section size: {section_size:.1f} frames, Blend width: {blend_width} frames, Method: {blend_method}"
        )
        visualization.append("")

        # Create frame markers
        markers = ["╶"] * num_frames

        # Mark section boundaries
        for i in range(1, num_prompts):
            boundary = int(i * section_size)
            if boundary < num_frames:
                markers[boundary] = "┃"  # Section boundary

        # Mark blend zones if requested
        if show_blend_zones and blend_width > 0:
            for i in range(1, num_prompts):
                boundary = int(i * section_size)
                # Convert latent blend_width to pixel space (multiply by 4)
                pixel_blend_width = blend_width
                zone_start = max(0, boundary - pixel_blend_width)
                zone_end = min(num_frames - 1, boundary + pixel_blend_width - 1)

                # Mark the blend zone
                for j in range(zone_start, zone_end + 1):
                    if j != boundary:  # Don't overwrite section boundary
                        # Calculate blend ratio
                        distance = abs(j - boundary)
                        raw_ratio = 1.0 - (distance / pixel_blend_width)

                        # Apply the selected curve
                        if blend_method == "smooth":
                            blend_ratio = raw_ratio * raw_ratio * (3 - 2 * raw_ratio)
                        elif blend_method == "ease_in":
                            blend_ratio = raw_ratio * raw_ratio
                        elif blend_method == "ease_out":
                            blend_ratio = raw_ratio * (2 - raw_ratio)
                        else:
                            blend_ratio = raw_ratio

                        # Use different characters based on blend ratio
                        if blend_ratio > 0.75:
                            markers[j] = "▓"
                        elif blend_ratio > 0.5:
                            markers[j] = "▒"
                        elif blend_ratio > 0.25:
                            markers[j] = "░"
                        else:
                            markers[j] = "·"

        # Create the visualization string
        vis_line = "".join(markers)
        chunk_size = 80

        # Split into chunks for readability
        for i in range(0, len(vis_line), chunk_size):
            chunk = vis_line[i : i + chunk_size]
            frame_start = i
            frame_end = min(i + chunk_size - 1, num_frames - 1)
            visualization.append(f"{frame_start:4d} {chunk} {frame_end:4d}")

        # Add legend
        visualization.append("")
        visualization.append("Legend:")
        visualization.append("┃ - Section boundary")
        if show_blend_zones:
            visualization.append("▓▒░· - Blend zone (▓=strongest blend)")
        visualization.append("╶ - Regular frame")

        return ("\n".join(visualization),)


# Node mapping for ComfyUI
NODE_CLASS_MAPPINGS = {
    "WanSmartBlend": WanSmartBlend,
    "WanBlendVisualize": WanBlendVisualize,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanSmartBlend": "Wan Smart Blend",
    "WanBlendVisualize": "Wan Blend Visualize",
}