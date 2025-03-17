# WanSeamlessFlow/visualization.py

from PIL import Image, ImageDraw, ImageFont
import torch
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
import math
from .blending import BlendFunctions

def pil_image_to_tensor(pil_image):
    """Convert PIL Image to torch tensor for ComfyUI (explicitly ensure uint8 and CHW format)."""
    image_np = np.array(pil_image, dtype=np.uint8)  # Ensure numpy array is uint8
    image_tensor = torch.from_numpy(image_np).float() / 255.0
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension: HWC -> BHWC
    return image_tensor


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


def calculate_embedding_distance(embed1: torch.Tensor, embed2: torch.Tensor) -> float:
    """
    Calculate semantic distance between two embeddings using cosine similarity.

    Args:
        embed1: First embedding tensor
        embed2: Second embedding tensor

    Returns:
        Float distance value (lower means more similar)
    """
    # Convert to float32 for consistent calculation
    e1 = embed1.float().mean(dim=0, keepdim=True)
    e2 = embed2.float().mean(dim=0, keepdim=True)

    # Calculate cosine similarity
    cos_sim = torch.nn.functional.cosine_similarity(e1, e2, dim=1)
    # Convert to distance (1 - similarity, so lower is more similar)
    distance = 1.0 - cos_sim.item()

    return distance


def create_distance_visualization(
    embeddings: List[torch.Tensor], blend_width: int, blend_method: str, num_frames: int
) -> Tuple[str, Dict[str, Any]]:
    """
    Create visualization of embedding distances and transition difficulty.

    Args:
        embeddings: List of prompt embeddings
        blend_width: Current blend width setting
        blend_method: Current blend method
        num_frames: Total number of frames

    Returns:
        HTML visualization and suggestions dictionary
    """
    if len(embeddings) <= 1:
        return "<p>Need at least 2 embeddings to analyze distances.</p>", {}

    # Calculate pairwise distances
    distances = []
    for i in range(len(embeddings) - 1):
        distance = calculate_embedding_distance(embeddings[i], embeddings[i + 1])
        distances.append(distance)

    # Normalize distances to a 0-1 scale for better visualization
    if len(distances) > 1:
        min_dist = min(distances)
        max_dist = max(distances)
        norm_range = max(0.1, max_dist - min_dist)  # Avoid division by zero
        norm_distances = [(d - min_dist) / norm_range for d in distances]
    else:
        norm_distances = [0.5]  # Default for single distance

    # Generate recommendations based on distances
    recommendations = generate_recommendations(
        distances, norm_distances, blend_width, blend_method, num_frames
    )

    # Create visualization HTML
    section_size = num_frames / len(embeddings)
    html = create_html_visualization(
        distances,
        norm_distances,
        embeddings,
        section_size,
        blend_width,
        recommendations,
    )

    return html, recommendations


def difficulty_to_color(difficulty: float) -> str:
    """Convert difficulty score to color."""
    # Green (easy) to red (hard) gradient
    r = min(255, int(difficulty * 255))
    g = min(255, int((1 - difficulty) * 255))
    b = 50
    return f"rgb({r}, {g}, {b})"


def generate_recommendations(
    distances: List[float],
    norm_distances: List[float],
    current_blend_width: int,
    current_blend_method: str,
    num_frames: int,
) -> Dict[str, Any]:
    """Generate recommendations based on embedding distances."""
    section_size = num_frames / (len(distances) + 1)

    # Overall recommendations
    recommendations = {
        "overall_difficulty": sum(norm_distances) / len(norm_distances)
        if norm_distances
        else 0,
        "hardest_transition": distances.index(max(distances)) if distances else 0,
        "easiest_transition": distances.index(min(distances)) if distances else 0,
        "per_transition": [],
    }

    # Per-transition recommendations
    for i, (dist, norm_dist) in enumerate(zip(distances, norm_distances)):
        # Determine recommended blend width based on distance
        # Higher distance needs wider blend
        rec_blend_width = max(4, min(32, int(norm_dist * 24) + 8))

        # Recommend blend method based on distance
        if norm_dist > 0.7:  # Very different prompts
            rec_method = "smooth"
        elif norm_dist > 0.4:  # Moderately different
            rec_method = "sine"
        else:  # Similar prompts
            rec_method = "linear"

        transition_rec = {
            "index": i,
            "distance": dist,
            "normalized_distance": norm_dist,
            "difficulty": "High"
            if norm_dist > 0.7
            else "Medium"
            if norm_dist > 0.4
            else "Low",
            "recommended_blend_width": rec_blend_width,
            "recommended_blend_method": rec_method,
            "frame_position": int((i + 1) * section_size),
        }

        recommendations["per_transition"].append(transition_rec)

    # Overall optimal settings
    if distances:
        avg_dist = sum(norm_distances) / len(norm_distances)
        recommendations["recommended_blend_method"] = (
            "smooth" if avg_dist > 0.6 else "sine" if avg_dist > 0.3 else "linear"
        )
        recommendations["recommended_blend_width"] = max(
            4, min(24, int(avg_dist * 20) + 8)
        )
        recommendations["recommended_transition_count"] = (
            3 if avg_dist > 0.7 else 2 if avg_dist > 0.4 else 1
        )

    return recommendations


def create_distance_graph_canvas(
    distances,
    total_frames,
    blend_frames,
    prompts,
    blend_method="linear",
    section_boundaries=None,
    suggested_settings=None,
    canvas_width=800,
    canvas_height=400,
):
    """
    Generates a canvas-based visualization of embedding distances, mimicking HTML style.
    """
    num_prompts = len(prompts)
    if num_prompts <= 1:
        return Image.new("RGB", (canvas_width, canvas_height), "white")

    image = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(image)
    font_small = ImageFont.load_default()
    font_large = (
        ImageFont.load_default()
    )  # Changed to load_default() - Use default font for large text too

    # --- 1. Distance Bars ---
    bar_area_height = canvas_height * 0.3  # Height for distance bars section
    bar_width = canvas_width / (num_prompts - 1) if num_prompts > 1 else 0
    max_distance = max(distances) if distances else 1.0
    bar_scale = bar_area_height / max_distance if max_distance > 0 else 0.0
    bar_start_y = canvas_height * 0.1  # Start Y for bars (top margin)

    for i, distance in enumerate(distances):
        bar_height = distance * bar_scale
        bar_x_start = i * bar_width
        bar_x_end = (i + 1) * bar_width
        normalized_distance = distance / max_distance if max_distance > 0 else 0.0
        diff_color = difficulty_to_color(
            normalized_distance
        )  # Get color from difficulty function

        # Draw distance bar
        draw.rectangle(
            [
                (bar_x_start, bar_start_y + bar_area_height - bar_height),
                (bar_x_end, bar_start_y + bar_area_height),
            ],
            fill=diff_color,
        )
        # Distance text on bar
        text_color = "white"  # White text on colored bar
        text_pos_x = bar_x_start + bar_width / 2
        text_pos_y = bar_start_y + bar_area_height - bar_height / 2
        draw.text(
            (text_pos_x, text_pos_y),
            f"{distance:.2f}",
            fill=text_color,
            anchor="mm",
            font=font_small,
        )
        # Prompt labels below bars
        draw.text(
            (text_pos_x, bar_start_y + bar_area_height + 15),
            f"P{i + 1}-P{i + 2}",
            fill="black",
            anchor="mt",
            font=font_small,
        )

    # --- 2. Frame Ruler ---
    ruler_y_start = bar_start_y + bar_area_height + 40  # Position ruler below bars
    ruler_height = 30
    draw.rectangle(
        [(50, ruler_y_start), (canvas_width - 50, ruler_y_start + ruler_height)],
        fill="#f0f0f0",
    )  # Ruler background

    section_size = total_frames / num_prompts if num_prompts > 1 else 0
    for i in range(num_prompts):
        marker_x = (
            50 + (i * section_size / total_frames) * (canvas_width - 100)
            if total_frames > 0
            else 50
        )  # Calculate marker X position
        if i > 0:  # Section boundaries markers (except for the first prompt)
            draw.line(
                [(marker_x, ruler_y_start), (marker_x, ruler_y_start + ruler_height)],
                fill="#333",
                width=2,
            )
        # Prompt labels on ruler
        draw.text(
            (marker_x, ruler_y_start + ruler_height + 5),
            f"P{i + 1}",
            fill="black",
            anchor="mt",
            font=font_small,
        )

        if i < num_prompts - 1 and blend_frames > 0:  # Blend Zones (between prompts)
            transition_pos_frame = (i + 1) * section_size
            blend_start_frame = max(0, transition_pos_frame - blend_frames)
            blend_end_frame = min(total_frames, transition_pos_frame + blend_frames)

            blend_start_x = (
                50 + (blend_start_frame / total_frames) * (canvas_width - 100)
                if total_frames > 0
                else 50
            )
            blend_end_x = (
                50 + (blend_end_frame / total_frames) * (canvas_width - 100)
                if total_frames > 0
                else 50
            )
            diff_color = difficulty_to_color(
                distances[i] / max_distance if max_distance > 0 else 0.5
            )  # Color from distance

            draw.rectangle(
                [
                    (blend_start_x, ruler_y_start),
                    (blend_end_x, ruler_y_start + ruler_height),
                ],
                fill=diff_color,
                outline="black",
                width=1,
            )  # Blend Zone rect

    # --- 3. Summary & Recommendations Text (Simple Text on Canvas for now) ---
    text_y_start = ruler_y_start + ruler_height + 30

    if suggested_settings:
        summary_text_lines = [
            "Summary:",
            f"Number of embeddings: {num_prompts}",
            f"Overall transition difficulty: {'High' if suggested_settings['overall_difficulty'] > 0.7 else 'Medium' if suggested_settings['overall_difficulty'] > 0.4 else 'Low'}",
            f"Hardest transition: P{suggested_settings['hardest_transition'] + 1}-P{suggested_settings['hardest_transition'] + 2} (Distance: {distances[suggested_settings['hardest_transition']]:.3f})",
            f"Easiest transition: P{suggested_settings['easiest_transition'] + 1}-P{suggested_settings['easiest_transition'] + 2} (Distance: {distances[suggested_settings['easiest_transition']]:.3f})",
            "",
            "Recommendations:",
            f"Recommended overall blend width: {suggested_settings['recommended_blend_width']} (current: {blend_frames})",
            f"Recommended blend method: {suggested_settings['recommended_blend_method']} (current: {blend_method})",
            f"Recommended transition count: {suggested_settings['recommended_transition_count']}",
            "",
            "Per-Transition Recommendations:",
        ]

        current_y = text_y_start
        for line in summary_text_lines:
            draw.text(
                (10, current_y),
                line,
                fill="black",
                font=font_large
                if "Summary" in line
                or "Recommendations" in line
                or "Per-Transition" in line
                else font_small,
                anchor="lt",
            )
            current_y += 20

        # Per-transition recs (simplified - just text)
        for rec in suggested_settings["per_transition"]:
            diff_color = difficulty_to_color(rec["normalized_distance"])
            rec_text = f"  P{rec['index'] + 1}-P{rec['index'] + 2}: Difficulty: {rec['difficulty']} (color={diff_color}), Rec. Width: {rec['recommended_blend_width']}, Method: {rec['recommended_blend_method']}"
            draw.text(
                (20, current_y), rec_text, fill="black", font=font_small, anchor="lt"
            )
            current_y += 15

    return image


def create_html_visualization(
    distances: List[float],
    norm_distances: List[float],
    embeddings: List[torch.Tensor],
    section_size: float,
    blend_width: int,
    recommendations: Dict[str, Any],
) -> str:
    """Create HTML visualization of embedding distances and transitions."""
    html = """
    <style>
        .container { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; }
        .distance-bar { height: 30px; margin: 5px 0; border-radius: 3px; display: flex; align-items: center; padding: 0 10px; color: white; font-weight: bold; }
        .frame-ruler { position: relative; height: 60px; margin: 20px 0; background: #f0f0f0; border-radius: 3px; }
        .frame-marker { position: absolute; width: 2px; background: #333; height: 100%; }
        .frame-label { position: absolute; top: -20px; transform: translateX(-50%); font-size: 12px; }
        .blend-zone { position: absolute; height: 100%; opacity: 0.5; border-radius: 3px; }
        .summary { background: #f5f5f5; padding: 15px; border-radius: 5px; margin: 20px 0; }
        .recommendations { background: #e9f7ef; padding: 15px; border-radius: 5px; margin: 20px 0; }
        .difficulty-label { padding: 3px 8px; border-radius: 10px; font-size: 12px; margin-left: 10px; color: white; }
    </style>
    <div class="container">
        <h2>Embedding Distance Analysis</h2>
        <p>This visualization shows the semantic distance between consecutive prompts and helps predict transition difficulty.</p>
        
        <div class="summary">
            <h3>Summary</h3>
            <p>Number of embeddings: <strong>{}</strong></p>
            <p>Overall transition difficulty: <strong>{}</strong></p>
            <p>Hardest transition: <strong>{} to {}</strong> (Distance: {:.3f})</p>
            <p>Easiest transition: <strong>{} to {}</strong> (Distance: {:.3f})</p>
        </div>
    """.format(
        len(embeddings),
        "High"
        if recommendations["overall_difficulty"] > 0.7
        else "Medium"
        if recommendations["overall_difficulty"] > 0.4
        else "Low",
        recommendations["hardest_transition"] + 1,
        recommendations["hardest_transition"] + 2,
        distances[recommendations["hardest_transition"]],
        recommendations["easiest_transition"] + 1,
        recommendations["easiest_transition"] + 2,
        distances[recommendations["easiest_transition"]],
    )

    # Distance bars visualization
    html += "<h3>Transition Distances</h3>"
    for i, (dist, norm_dist) in enumerate(zip(distances, norm_distances)):
        difficulty = (
            "High" if norm_dist > 0.7 else "Medium" if norm_dist > 0.4 else "Low"
        )
        diff_color = difficulty_to_color(norm_dist)

        html += f"""
        <div>
            <p>Prompt {i + 1} to {i + 2} <span class="difficulty-label" style="background-color: {diff_color};">{difficulty}</span></p>
            <div class="distance-bar" style="width: {int(norm_dist * 100)}%; background-color: {diff_color};">
                {dist:.3f}
            </div>
        </div>
        """

    # Frame position visualization
    html += "<h3>Frame Position & Blend Zones</h3>"
    html += "<div class='frame-ruler'>"

    # Add markers for each embedding
    for i in range(len(embeddings)):
        position = i * section_size
        position_percent = (position / (len(embeddings) * section_size)) * 100
        html += f"""
        <div class="frame-marker" style="left: {position_percent}%;">
            <div class="frame-label">Prompt {i + 1}</div>
        </div>
        """

        # Add blend zones except for the last prompt
        if i < len(embeddings) - 1:
            # Calculate transition position
            transition_pos = (i + 1) * section_size
            transition_percent = (
                transition_pos / (len(embeddings) * section_size)
            ) * 100

            # Calculate blend zone
            blend_start = transition_pos - blend_width
            blend_end = transition_pos + blend_width

            blend_start_percent = (blend_start / (len(embeddings) * section_size)) * 100
            blend_width_percent = (
                (blend_end - blend_start) / (len(embeddings) * section_size)
            ) * 100

            # Get difficulty color for this transition
            diff_color = difficulty_to_color(norm_distances[i])

            html += f"""
            <div class="blend-zone" style="left: {blend_start_percent}%; width: {blend_width_percent}%; background-color: {diff_color};">
            </div>
            """

    html += "</div>"

    # Recommendations
    html += """
    <div class="recommendations">
        <h3>Recommendations</h3>
        <p>Based on the embedding distances, here are suggested settings:</p>
        <ul>
            <li>Recommended overall blend width: <strong>{}</strong> (currently {})</li>
            <li>Recommended blend method: <strong>{}</strong> (currently {})</li>
            <li>Recommended transition count: <strong>{}</strong></li>
        </ul>
        
        <h4>Per-Transition Recommendations</h4>
        <table style="width: 100%; border-collapse: collapse; margin-top: 10px;">
            <tr style="background: #ddd;">
                <th style="padding: 8px; text-align: left;">Transition</th>
                <th style="padding: 8px; text-align: left;">Difficulty</th>
                <th style="padding: 8px; text-align: left;">Recommended Width</th>
                <th style="padding: 8px; text-align: left;">Recommended Method</th>
            </tr>
    """.format(
        recommendations["recommended_blend_width"],
        blend_width,
        recommendations["recommended_blend_method"],
        blend_method,
        recommendations["recommended_transition_count"],
    )

    # Add per-transition recommendations
    for rec in recommendations["per_transition"]:
        diff_color = difficulty_to_color(rec["normalized_distance"])
        html += f"""
        <tr>
            <td style="padding: 8px; border-bottom: 1px solid #ddd;">Prompt {rec["index"] + 1} to {rec["index"] + 2}</td>
            <td style="padding: 8px; border-bottom: 1px solid #ddd;">
                <span class="difficulty-label" style="background-color: {diff_color};">{rec["difficulty"]}</span>
            </td>
            <td style="padding: 8px; border-bottom: 1px solid #ddd;">{rec["recommended_blend_width"]}</td>
            <td style="padding: 8px; border-bottom: 1px solid #ddd;">{rec["recommended_blend_method"]}</td>
        </tr>
        """

    html += """
        </table>
    </div>
    """

    html += "</div>"
    return html