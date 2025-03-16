# WanSeamlessFlow/visualization.py

from typing import Optional, List
from .blending import BlendFunctions

def create_transition_visualization(
    num_frames: int,
    num_prompts: int,
    blend_width: int,
    blend_method: str = "linear",
    show_blend_zones: bool = True,
    frame_markers: Optional[List[str]] = None
) -> str:
    """
    Create a textual visualization of prompt transitions.
    
    Args:
        num_frames: Total number of frames
        num_prompts: Number of prompts
        blend_width: Width of transition zone
        blend_method: Type of interpolation curve
        show_blend_zones: Whether to highlight blend zones
        frame_markers: Custom frame markers
        
    Returns:
        String visualization
    """
    # Use default frame markers if not provided
    if frame_markers is None:
        frame_markers = ["╶"] * num_frames
    
    # Nothing to visualize for edge cases
    if blend_width == 0 or num_prompts <= 1:
        return "No blend zones configured or only one prompt available."
    
    # Calculate section size
    section_size = num_frames / num_prompts
    
    # Build visualization
    visualization = []
    visualization.append(f"Transition visualization for {num_frames} frames with {num_prompts} prompts")
    visualization.append(f"Section size: {section_size:.1f} frames, Blend width: {blend_width} frames, Method: {blend_method}")
    visualization.append("")
    
    # Mark section boundaries
    for i in range(1, num_prompts):
        boundary = int(i * section_size)
        if boundary < num_frames:
            frame_markers[boundary] = "┃"  # Section boundary
    
    # Mark blend zones if requested
    if show_blend_zones and blend_width > 0:
        # Get the blend function
        blend_func = getattr(BlendFunctions, blend_method, BlendFunctions.linear)
        
        for i in range(1, num_prompts):
            boundary = int(i * section_size)
            zone_start = max(0, boundary - blend_width)
            zone_end = min(num_frames-1, boundary + blend_width - 1)
            
            # Mark the blend zone
            for j in range(zone_start, zone_end + 1):
                if j != boundary:  # Don't overwrite section boundary
                    # Calculate raw blend ratio
                    distance = abs(j - boundary)
                    raw_ratio = 1.0 - (distance / blend_width)
                    
                    # Apply the selected curve
                    blend_ratio = blend_func(raw_ratio)
                    
                    # Use different characters based on blend ratio
                    if blend_ratio > 0.75:
                        frame_markers[j] = "▓"
                    elif blend_ratio > 0.5:
                        frame_markers[j] = "▒"
                    elif blend_ratio > 0.25:
                        frame_markers[j] = "░"
                    else:
                        frame_markers[j] = "·"
    
    # Create the visualization string
    vis_line = "".join(frame_markers)
    chunk_size = 80
    
    # Split into chunks for readability
    for i in range(0, len(vis_line), chunk_size):
        chunk = vis_line[i:i+chunk_size]
        frame_start = i
        frame_end = min(i+chunk_size-1, num_frames-1)
        visualization.append(f"{frame_start:4d} {chunk} {frame_end:4d}")
    
    # Add legend
    visualization.append("")
    visualization.append("Legend:")
    visualization.append("┃ - Section boundary")
    if show_blend_zones:
        visualization.append("▓▒░· - Blend zone (▓=strongest blend)")
    visualization.append("╶ - Regular frame")
    
    return "\n".join(visualization)