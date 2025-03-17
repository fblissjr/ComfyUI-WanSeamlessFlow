# WanSeamlessFlow/nodes.py

from .blending import blend_embeddings, harmonize_embeddings, BlendFunctions
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
from .wan_adaptive_flow import WanAdaptiveFlow
from .visualization import (
    create_distance_graph_canvas,
    generate_recommendations,
    calculate_embedding_distance,
    pil_image_to_tensor,
)
from PIL import Image


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
                    [
                        "linear",
                        "smooth",
                        "ease_in",
                        "ease_out",
                        "sine",
                        "circ",
                        "bounce",
                    ],
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

            # Add num_frames and texts even for single prompt case for consistency
            if "texts" in text_embeds:
                result["texts"] = text_embeds["texts"]
                log.info("WanSmartBlend: Added texts to output.")

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
                result["blend_width"] = 0  # Disable blending
                result["prompt_embeds"] = text_embeds[
                    "prompt_embeds"
                ]  # Restore original embeddings

        # After the harmonization check for issues
        if not result["prompt_embeds"] or len(result["prompt_embeds"]) == 0:
            log.info(
                "WARNING: Harmonization resulted in empty prompt array, restoring original"
            )
            result["blend_width"] = 0  # Disable blending
            result["prompt_embeds"] = text_embeds[
                "prompt_embeds"
            ]  # Restore original embeddings

        # Check embedding shape consistency (after attempted harmonization)
        shapes = [embed.shape for embed in result["prompt_embeds"]]
        if len(set(str(s) for s in shapes)) > 1:
            if verbosity > 0:
                log.info(
                    f"WanSmartBlend: Detected varying embedding shapes AFTER harmonization: {shapes}. Blending disabled."
                )
            result["blend_width"] = (
                0  # Disable blending if shapes are still inconsistent
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

        # Pass along num_frames and texts from input text_embeds
        if "num_frames" in text_embeds:
            result["num_frames"] = text_embeds["num_frames"]
            log.info("WanSmartBlend: Added num_frames to output.")
        else:
            log.info("WanSmartBlend: num_frames not found in input text_embeds.")

        if "texts" in text_embeds:
            result["texts"] = text_embeds["texts"]
            log.info("WanSmartBlend: Added texts to output.")
        else:
            log.info("WanSmartBlend: texts not found in input text_embeds.")

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


# throttlekitty
class WanEmbeddingPrevizCanvas:
    """
    Analyzes text embeddings to visualize transitions and predict difficulty using Canvas,
    taking inputs directly from upstream nodes for prompts, frames, and blend method.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text_embeds": ("WANVIDEOTEXTEMBEDS",),
            },
            "optional": {  # Make these inputs optional
                "num_frames": (
                    "INT",
                    {  # Optional num_frames input
                        "default": 81,
                        "min": 1,
                        "max": 1000,
                        "step": 1,
                        "tooltip": "Total number of frames in the video (optional, will try to get from Empty Embeds or text_embeds if not connected)",
                    },
                ),
                "blend_width": (
                    "INT",
                    {  # Optional blend_width input
                        "default": 8,
                        "min": 0,
                        "max": 32,
                        "step": 1,
                        "tooltip": "Current blend width setting (optional, will try to get from Smart Blend or text_embeds if not connected)",
                    },
                ),
                "blend_method": (
                    [
                        "linear",
                        "smooth",
                        "ease_in",
                        "ease_out",
                        "sine",
                        "circ",
                        "bounce",
                    ],
                    {  # Optional blend_method input
                        "default": "smooth",
                        "tooltip": "Current blend method setting (optional, will try to get from Smart Blend or text_embeds if not connected)",
                    },
                ),
                "prompts": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "Prompt 1\nPrompt 2\nPrompt 3",
                        "tooltip": "Prompts (optional, will try to get from Granular Text Encode or text_embeds if not connected)",
                    },
                ),  # Optional prompts input
                "optimize_order": (
                    "BOOLEAN",
                    {"default": True},
                ),  # Keep optimize_order if needed
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("visualization", "recommendations")
    FUNCTION = "analyze_canvas"
    CATEGORY = "WanSeamlessFlow"
    DESCRIPTION = "Analyzes embeddings, visualizes transitions on Canvas, predicts difficulty, and suggests settings. Takes inputs from upstream nodes where possible."

    def analyze_canvas(
        self,
        text_embeds,
        num_frames=None,
        blend_width=None,
        blend_method=None,
        prompts=None,
        optimize_order=True,
    ):  # Make inputs optional in function signature too
        """
        Analyze embeddings, generate Canvas visualization, and provide recommendations,
        getting inputs from upstream nodes if connected, or using manual inputs if provided.
        """

        embeddings = text_embeds.get("prompt_embeds", [])

        if not embeddings or len(embeddings) < 2:
            blank_image = Image.new("RGB", (800, 400), "white")
            image_tensor = pil_image_to_tensor(blank_image)
            return (image_tensor,), "Need at least 2 embeddings to analyze transitions."

        # --- Get num_frames ---
        if (
            num_frames is None
        ):  # Try to get num_frames from text_embeds if not provided manually
            num_frames = text_embeds.get(
                "num_frames"
            )  # Check if num_frames is in text_embeds
            if num_frames is None:
                num_frames_from_empty = text_embeds.get(
                    "num_frames_source"
                )  # Check if WanVideoEmptyEmbeds passed num_frames via "num_frames_source"
                if num_frames_from_empty:
                    num_frames = num_frames_from_empty
                else:
                    num_frames = 81  # Default if still not found
                    log.info(
                        "WanEmbeddingPrevizCanvas: Warning - num_frames not found from input, using default."
                    )
        else:
            # Manual num_frames input takes precedence, but also pass it along in text_embeds for potential downstream use if it's not already there
            if "num_frames" not in text_embeds:
                text_embeds["num_frames"] = num_frames

        # --- Get blend_width and blend_method ---
        if (
            blend_width is None
        ):  # Try to get blend_width from text_embeds if not provided manually
            blend_width = text_embeds.get("blend_width")
            if blend_width is None:
                blend_width = 8  # Default if still not found
                log.info(
                    "WanEmbeddingPrevizCanvas: Warning - blend_width not found from input, using default."
                )

        if (
            blend_method is None
        ):  # Try to get blend_method from text_embeds if not provided manually
            blend_method = text_embeds.get("blend_method")
            if blend_method is None:
                blend_method = "smooth"  # Default if still not found
                log.info(
                    "WanEmbeddingPrevizCanvas: Warning - blend_method not found from input, using default."
                )

        # --- Get prompts ---
        if prompts is None:  # If prompts are not manually provided
            # Try to extract prompts from text_embeds (assuming WanVideoGranularTextEncode or similar stores them)
            input_texts = text_embeds.get(
                "texts"
            )  # Assuming 'texts' key holds original prompt strings
            if input_texts:
                prompts = "\n".join(
                    input_texts
                )  # Join list of prompts into multiline string
            else:
                prompts = "Prompt 1\nPrompt 2\nPrompt 3"  # Fallback default prompts
                log.info(
                    "WanEmbeddingPrevizCanvas: Warning - prompts not found from input, using default."
                )

        distances = []
        for i in range(len(embeddings) - 1):
            distance = calculate_embedding_distance(embeddings[i], embeddings[i + 1])
            distances.append(distance)

        norm_distances = []
        if len(distances) > 1:
            min_dist = min(distances)
            max_dist = max(distances)
            norm_range = max(0.1, max_dist - min_dist)
            norm_distances = [(d - min_dist) / norm_range for d in distances]
        elif distances:
            norm_distances = [0.5]
        else:
            norm_distances = []

        prompts_list = prompts.strip().split("\n")  # Split multiline prompts into list
        recommendations = generate_recommendations(
            distances, norm_distances, blend_width, blend_method, num_frames
        )

        graph_image = create_distance_graph_canvas(
            distances,
            num_frames,
            blend_width,
            prompts_list,
            blend_method,
            suggested_settings=recommendations,
        )
        image_tensor = pil_image_to_tensor(graph_image)

        rec_str = "Embedding Analysis Recommendations (Canvas Visualization):\n\n"
        rec_str += (
            f"Overall difficulty: {recommendations.get('overall_difficulty', 0):.2f}\n"
        )
        rec_str += f"Recommended blend width: {recommendations.get('recommended_blend_width', blend_width)}\n"
        rec_str += f"Recommended blend method: {recommendations.get('recommended_blend_method', blend_method)}\n"
        rec_str += f"Recommended transition count: {recommendations.get('recommended_transition_count', 1)}\n\n"

        # Add per-transition recommendations
        rec_str += "Per Transition Recommendations:\n"
        for rec in recommendations.get("per_transition", []):
            rec_str += f"Prompt {rec['index'] + 1} to {rec['index'] + 2}:\n"
            rec_str += f"  - Difficulty: {rec['difficulty']} (distance: {rec['distance']:.3f})\n"
            rec_str += f"  - Recommended width: {rec['recommended_blend_width']}\n"
            rec_str += f"  - Recommended method: {rec['recommended_blend_method']}\n"
            rec_str += f"  - Frame position: ~{rec['frame_position']}\n\n"

        return (image_tensor,), rec_str


class WanEmbeddingPreviz:
    """
    Analyzes text embeddings to visualize transitions and predict difficulty.
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
                "blend_width": (
                    "INT",
                    {
                        "default": 8,
                        "min": 0,
                        "max": 32,
                        "step": 1,
                        "tooltip": "Current blend width setting",
                    },
                ),
                "blend_method": (
                    ["linear", "smooth", "ease_in", "ease_out", "sine"],
                    {
                        "default": "smooth",
                        "tooltip": "Current blend method setting",
                    },
                ),
            }
        }

    RETURN_TYPES = (
        "HTML",
        "STRING",
    )
    RETURN_NAMES = (
        "visualization",
        "recommendations",
    )
    FUNCTION = "analyze"
    CATEGORY = "WanSeamlessFlow"
    DESCRIPTION = "Analyzes text embeddings to visualize transitions, predict difficulty, and suggest optimal settings."

    def analyze(self, text_embeds, num_frames, blend_width, blend_method):
        """
        Analyze embeddings to visualize transitions and predict difficulty.

        Args:
            text_embeds: Dictionary containing prompt embeddings
            num_frames: Total number of frames in the video
            blend_width: Current blend width setting
            blend_method: Current blend method setting

        Returns:
            HTML visualization and recommendations as string
        """
        from .visualization import create_distance_visualization
        import json

        # Extract embeddings from text_embeds
        embeddings = text_embeds.get("prompt_embeds", [])

        if not embeddings or len(embeddings) < 2:
            html = "<p>Error: Need at least 2 embeddings to analyze transitions.</p>"
            return (html, "No recommendations available (need at least 2 embeddings)")

        # Generate visualization and recommendations
        html, recommendations = create_distance_visualization(
            embeddings, blend_width, blend_method, num_frames
        )

        # Format recommendations as string
        rec_str = "Embedding Analysis Recommendations:\n\n"
        rec_str += (
            f"Overall difficulty: {recommendations.get('overall_difficulty', 0):.2f}\n"
        )
        rec_str += f"Recommended blend width: {recommendations.get('recommended_blend_width', blend_width)}\n"
        rec_str += f"Recommended blend method: {recommendations.get('recommended_blend_method', blend_method)}\n"
        rec_str += f"Recommended transition count: {recommendations.get('recommended_transition_count', 1)}\n\n"

        # Add per-transition recommendations
        rec_str += "Per Transition Recommendations:\n"
        for rec in recommendations.get("per_transition", []):
            rec_str += f"Prompt {rec['index'] + 1} to {rec['index'] + 2}:\n"
            rec_str += f"  - Difficulty: {rec['difficulty']} (distance: {rec['distance']:.3f})\n"
            rec_str += f"  - Recommended width: {rec['recommended_blend_width']}\n"
            rec_str += f"  - Recommended method: {rec['recommended_blend_method']}\n"
            rec_str += f"  - Frame position: ~{rec['frame_position']}\n\n"

        return (html, rec_str)


def pil_image_to_tensor_minimal(pil_image):  # Minimal tensor conversion for test node
    """Convert PIL Image to torch tensor for ComfyUI - minimal version."""
    image_np = np.array(pil_image, dtype=np.uint8)
    image_tensor = torch.from_numpy(image_np).float() / 255.0
    image_tensor = image_tensor.unsqueeze(
        0
    )  # Just BHWC - Removed permute for Channels-First
    return image_tensor


class WanMinimalCanvasTest:
    """
    Extremely minimal Canvas test node - just outputs a blank white canvas.
    ACCEPTS text_embeds input for testing dictionary influence.
    For debugging TypeError.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "width": ("INT", {"default": 800, "min": 64, "max": 2048}),
                "height": ("INT", {"default": 400, "min": 64, "max": 2048}),
                "text_embeds": (
                    "WANVIDEOTEXTEMBEDS",
                ),  # ADDED text_embeds input (optional)
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("canvas",)
    FUNCTION = "process"
    CATEGORY = "WanSeamlessFlow/Debug"  # Put in a Debug category

    def process(
        self, width, height, text_embeds
    ):  # Added text_embeds to process function (but we will ignore it)
        """Process method for minimal canvas test - now accepts text_embeds input."""
        blank_image = Image.new(
            "RGB", (width, height), "white"
        )  # Simple blank white image
        image_tensor = pil_image_to_tensor_minimal(
            blank_image
        )  # Use minimal tensor conversion
        return (image_tensor,)


# Node mapping for ComfyUI
NODE_CLASS_MAPPINGS = {
    "WanSmartBlend": WanSmartBlend,
    "WanBlendVisualize": WanBlendVisualize,
    "WanAdaptiveFlow": WanAdaptiveFlow,
    "WanEmbeddingPrevizCanvas": WanEmbeddingPrevizCanvas,
    "WanMinimalCanvasTest": WanMinimalCanvasTest,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanSmartBlend": "Wan Smart Blend",
    "WanBlendVisualize": "Wan Blend Visualize",
    "WanAdaptiveFlow": "Wan Adaptive Flow",
    "WanEmbeddingPreviz": "Wan Embedding Previz (Canvas)",
    "WanMinimalCanvasTest": "Wan Minimal Canvas Test",
}