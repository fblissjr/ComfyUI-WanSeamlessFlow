import torch
import numpy as np
import os
import logging
import comfy.model_management as mm
from comfy.utils import common_upscale
import traceback

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)

class WanAdaptiveFlow:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text_embeds": ("WANVIDEOTEXTEMBEDS",),
                "analysis_points": ("INT", {"default": 3, "min": 1, "max": 10, "step": 1, 
                                   "tooltip": "Number of points to analyze per transition"}),
                "first_pass_steps": ("INT", {"default": 8, "min": 1, "max": 30, "step": 1,
                                   "tooltip": "Number of steps for initial generation pass"}),
                "bridge_mode": (["Automatic", "Content-focused", "Lighting-focused", "Composition-focused"], 
                              {"default": "Automatic", "tooltip": "Type of bridge prompt to create"}),
                "bridge_count": ("INT", {"default": 1, "min": 1, "max": 5, "step": 1,
                               "tooltip": "Number of bridge prompts per transition"}),
                "bridge_strength": ("FLOAT", {"default": 0.75, "min": 0.1, "max": 1.0, "step": 0.05,
                                  "tooltip": "How strongly to apply bridge modifications"}),
            },
            "optional": {
                "model": ("WANVIDEOMODEL", {"tooltip": "WanVideo model for initial pass"}),
                "clip_vision": ("CLIP_VISION", {"tooltip": "CLIP Vision for frame analysis"}),
                "vae": ("WANVAE", {"tooltip": "VAE for decoding latents"}),
                "image_embeds": ("WANVIDIMAGE_EMBEDS", {"tooltip": "Image embeddings for initial pass"}),
                "t5_encoder": ("WANTEXTENCODER", {"tooltip": "T5 encoder for bridge prompts"}),
                "lighting_weight": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05}),
                "content_weight": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05}),
                "composition_weight": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.05}),
                "save_first_pass": ("BOOLEAN", {"default": True}),
                "original_prompts": ("STRING", {"multiline": True, "tooltip": "Original prompts separated by |"}),
            }
        }

    RETURN_TYPES = ("WANVIDEOTEXTEMBEDS", "STRING",)
    RETURN_NAMES = ("text_embeds", "bridge_info",)
    FUNCTION = "process"
    CATEGORY = "WanSeamlessFlow"
    DESCRIPTION = "Creates adaptive multi-pass video generation with improved transitions"

    def process(self, text_embeds, analysis_points, first_pass_steps, bridge_mode, bridge_count, bridge_strength,
                model=None, clip_vision=None, vae=None, image_embeds=None, t5_encoder=None,
                lighting_weight=0.8, content_weight=0.7, composition_weight=0.6, 
                save_first_pass=True, original_prompts=None):
        """
        Implements the multi-pass adaptive video generation process.
        """
        # Extract original prompts if provided
        prompt_list = []
        if original_prompts:
            prompt_list = [p.strip() for p in original_prompts.split('|')]
        
        # Check if we can run an initial pass
        can_run_initial_pass = (model is not None and image_embeds is not None)
        can_analyze_frames = (can_run_initial_pass and vae is not None and clip_vision is not None)
        can_encode_bridges = (t5_encoder is not None)
        
        # If we can't run initial pass, fallback to embedding interpolation
        if not can_run_initial_pass:
            log.info("WanAdaptiveFlow: Cannot run initial pass, using embedding interpolation")
            enhanced_embeds = self._interpolate_embeddings(
                text_embeds, bridge_count, bridge_strength
            )
            bridge_info = "Created transitions using embedding interpolation (no initial pass)"
            return (enhanced_embeds, bridge_info)
        
        # Run initial pass
        log.info("WanAdaptiveFlow: Running initial generation pass...")
        
        # We need to call the WanVideoSampler directly
        # This is a bit tricky since it's a class, not just a function
        try:
            initial_latents = self._run_initial_pass(model, text_embeds, image_embeds, first_pass_steps)
        except Exception as e:
            log.error(f"WanAdaptiveFlow: Error during initial pass: {str(e)}")
            log.error(traceback.format_exc())
            # Fallback to interpolation
            enhanced_embeds = self._interpolate_embeddings(
                text_embeds, bridge_count, bridge_strength
            )
            bridge_info = f"Error during initial pass: {str(e)}\nFallback to embedding interpolation"
            return (enhanced_embeds, bridge_info)
        
        # Generate bridge prompts
        bridge_prompts = []
        if can_analyze_frames:
            log.info("WanAdaptiveFlow: Analyzing transition frames...")
            
            # Identify transition points
            transition_frames = self._identify_transition_frames(
                text_embeds, initial_latents.shape[2], analysis_points
            )
            
            # Extract and analyze key frames
            try:
                frame_analyses = self._analyze_frames(
                    initial_latents, transition_frames, vae, clip_vision
                )
                
                # Generate bridge prompts
                bridge_prompts = self._create_bridge_prompts(
                    text_embeds, frame_analyses, bridge_mode, 
                    lighting_weight, content_weight, composition_weight,
                    prompt_list
                )
            except Exception as e:
                log.error(f"WanAdaptiveFlow: Error during frame analysis: {str(e)}")
                log.error(traceback.format_exc())
                # Continue with empty bridge prompts
        
        # Debug info about bridges
        bridge_info = "Bridge prompts:\n"
        for bridge in bridge_prompts:
            bridge_info += f"After prompt {bridge['transition_idx']}: {bridge['prompt']}\n"
        
        # Handle embedding generation
        enhanced_embeds = None
        
        if can_encode_bridges and bridge_prompts:
            log.info("WanAdaptiveFlow: Encoding bridge prompts...")
            try:
                enhanced_embeds = self._encode_bridge_prompts(
                    text_embeds, bridge_prompts, bridge_count, bridge_strength, t5_encoder
                )
                bridge_info += "\nEncoded bridge prompts successfully"
            except Exception as e:
                log.error(f"WanAdaptiveFlow: Error encoding bridge prompts: {str(e)}")
                log.error(traceback.format_exc())
                # Fallback to interpolation
                enhanced_embeds = None
        
        # If we couldn't encode bridges, use interpolation
        if enhanced_embeds is None:
            log.info("WanAdaptiveFlow: Using embedding interpolation")
            enhanced_embeds = self._interpolate_embeddings(
                text_embeds, bridge_count, bridge_strength
            )
            bridge_info += "\nNote: Used embedding interpolation for transitions"
        
        # Add blend information for WanVideoSampler
        enhanced_embeds["blend_width"] = text_embeds.get("blend_width", 8)
        enhanced_embeds["blend_method"] = text_embeds.get("blend_method", "smooth")
        
        return (enhanced_embeds, bridge_info)

    def _run_initial_pass(self, model, text_embeds, image_embeds, steps):
        """
        Run an initial low-quality generation pass.
        
        Args:
            model: WanVideo model
            text_embeds: Original text embeddings
            image_embeds: Image embeddings
            steps: Number of steps for initial pass
            
        Returns:
            Initial latents for analysis
        """
        # Import here to avoid circular imports
        from .nodes import WanVideoSampler
        
        # Create a sampler instance
        sampler = WanVideoSampler()
        
        # Run with reduced steps
        result = sampler.process(
            model=model,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            steps=steps,
            cfg=6.0,  # Use default cfg
            shift=5.0,  # Use default shift
            seed=12345,  # Fixed seed for consistency
            force_offload=True,
            scheduler="euler",  # Faster scheduler for initial pass
        )
        
        return result[0]["samples"]
    
    def _identify_transition_frames(self, text_embeds, num_frames, analysis_points):
        """
        Identify which frames to analyze at transition points.
        
        Args:
            text_embeds: Original text embeddings
            num_frames: Total number of frames (in latent space)
            analysis_points: Number of points to analyze per transition
            
        Returns:
            List of frame indices to analyze
        """
        # Count the number of prompts
        num_prompts = len(text_embeds["prompt_embeds"])
        
        # If only one prompt, nothing to analyze
        if num_prompts <= 1:
            return []
        
        # Calculate section size (frames per prompt)
        section_size = num_frames / num_prompts
        
        # For each transition, identify frames to analyze
        transition_frames = []
        for i in range(1, num_prompts):
            # Calculate the boundary frame
            boundary = int(i * section_size)
            
            # Add frames around the boundary
            for offset in range(-analysis_points//2, analysis_points//2 + 1):
                frame_idx = boundary + offset
                if 0 <= frame_idx < num_frames:
                    transition_frames.append(frame_idx)
        
        return transition_frames
    
    def _analyze_frames(self, initial_latents, transition_frames, vae, clip_vision):
        """
        Extract and analyze key frames using CLIP Vision.
        
        Args:
            initial_latents: Latents from initial pass
            transition_frames: List of frame indices to analyze
            vae: VAE for decoding
            clip_vision: CLIP Vision model for analysis
            
        Returns:
            Dictionary of frame analyses
        """
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        
        # Move VAE to device
        vae.to(device)
        
        # Move CLIP Vision to device
        clip_vision.model.to(device)
        
        frame_analyses = {}
        
        # For each transition frame, decode and analyze
        for frame_idx in transition_frames:
            # Extract the latent
            frame_latent = initial_latents.clone()[:, :, frame_idx:frame_idx+1, :, :]
            
            # Decode to image
            with torch.no_grad():
                decoded_image = vae.decode(frame_latent, device=device)[0]
            
            # Preprocess for CLIP
            image = decoded_image.permute(0, 2, 3, 1)  # [b,c,h,w] -> [b,h,w,c]
            image = (image - image.min()) / (image.max() - image.min())
            image = torch.clamp(image, 0.0, 1.0)
            
            # Process with CLIP Vision
            with torch.no_grad():
                if hasattr(clip_vision, "encode_image"):
                    # Standard CLIP interface
                    clip_features = clip_vision.encode_image(image.to(device))
                else:
                    # WanVideo CLIP interface
                    pixel_values = self._clip_preprocess(image.to(device))
                    clip_features = clip_vision.visual(pixel_values)
            
            # Store analysis
            frame_analyses[frame_idx] = {
                "features": clip_features.detach().cpu(),
                "position": frame_idx / initial_latents.shape[2]  # Normalized position
            }
        
        # Clean up
        vae.to(offload_device)
        clip_vision.model.to(offload_device)
        mm.soft_empty_cache()
        
        return frame_analyses
    
    def _clip_preprocess(self, image, size=224):
        """Preprocess images for CLIP Vision"""
        import torch
        import torch.nn.functional as F
        
        # Default normalization constants for CLIP
        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]
        
        # If 4D tensor [B,H,W,C], convert to [B,C,H,W]
        if image.dim() == 4 and image.shape[3] == 3:
            image = image.permute(0, 3, 1, 2)
        
        # Resize to expected CLIP input size
        if image.shape[2] != size or image.shape[3] != size:
            image = F.interpolate(image, size=(size, size), mode='bicubic', align_corners=False)
        
        # Normalize
        mean = torch.tensor(mean, device=image.device).view(1, 3, 1, 1)
        std = torch.tensor(std, device=image.device).view(1, 3, 1, 1)
        image = (image - mean) / std
        
        return image
    
    def _create_bridge_prompts(self, text_embeds, frame_analyses, bridge_mode, 
                             lighting_weight, content_weight, composition_weight,
                             prompt_list=None):
        """
        Generate bridge prompts based on frame analyses.
        
        Args:
            text_embeds: Original text embeddings
            frame_analyses: Frame analyses from CLIP Vision
            bridge_mode: Type of bridge to create
            lighting_weight: Weight for lighting aspects
            content_weight: Weight for content aspects
            composition_weight: Weight for composition aspects
            prompt_list: Original prompts if available
            
        Returns:
            List of bridge prompts
        """
        # Use provided prompts if available
        if prompt_list and len(prompt_list) > 0:
            original_prompts = prompt_list
        else:
            # Try to infer from the first embedding's shape
            num_prompts = len(text_embeds["prompt_embeds"])
            original_prompts = [f"Prompt {i}" for i in range(num_prompts)]
            log.warning(f"No original prompts available. Using placeholders. Embedding count: {num_prompts}")
        
        # If only one prompt, nothing to bridge
        if len(original_prompts) <= 1:
            return []
        
        # Group frame analyses by transition
        num_prompts = len(original_prompts)
        transitions = {}
        
        for frame_idx, analysis in frame_analyses.items():
            # Determine which transition this frame belongs to
            transition_idx = int(np.floor(analysis["position"] * num_prompts))
            if transition_idx >= num_prompts - 1:
                transition_idx = num_prompts - 2  # Ensure valid index
            
            # Add to the appropriate transition
            if transition_idx not in transitions:
                transitions[transition_idx] = []
            transitions[transition_idx].append(analysis)
        
        # Generate bridge prompts for each transition
        bridge_prompts = []
        
        for transition_idx, analyses in transitions.items():
            # Get the source and destination prompts
            source_prompt = original_prompts[transition_idx]
            dest_prompt = original_prompts[transition_idx + 1]
            
            # Choose bridge template based on mode
            if bridge_mode == "Content-focused":
                template = self._content_bridge_template
            elif bridge_mode == "Lighting-focused":
                template = self._lighting_bridge_template
            elif bridge_mode == "Composition-focused":
                template = self._composition_bridge_template
            else:  # Automatic
                # Determine what's changing most between prompts
                if self._is_lighting_change(source_prompt, dest_prompt):
                    template = self._lighting_bridge_template
                elif self._is_composition_change(source_prompt, dest_prompt):
                    template = self._composition_bridge_template
                else:
                    template = self._content_bridge_template
            
            # Apply the template
            bridge_prompt = template(
                source_prompt, dest_prompt, 
                analyses, lighting_weight, content_weight, composition_weight
            )
            
            bridge_prompts.append({
                "transition_idx": transition_idx,
                "prompt": bridge_prompt
            })
        
        return bridge_prompts
    
    def _is_lighting_change(self, source_prompt, dest_prompt):
        """Detect if the main change is lighting"""
        lighting_keywords = ["light", "dark", "shadow", "bright", "dim", "illuminat", 
                             "glow", "shine", "sunset", "dawn", "dusk", "morning", "night"]
        
        # Count lighting terms in both prompts
        source_count = sum(1 for keyword in lighting_keywords if keyword.lower() in source_prompt.lower())
        dest_count = sum(1 for keyword in lighting_keywords if keyword.lower() in dest_prompt.lower())
        
        # If either has significant lighting terms, consider it a lighting change
        return source_count >= 2 or dest_count >= 2
    
    def _is_composition_change(self, source_prompt, dest_prompt):
        """Detect if the main change is composition"""
        composition_keywords = ["composition", "angle", "view", "shot", "perspective", 
                              "zoom", "wide", "close", "distant", "framing", "orientation"]
        
        # Count composition terms in both prompts
        source_count = sum(1 for keyword in composition_keywords if keyword.lower() in source_prompt.lower())
        dest_count = sum(1 for keyword in composition_keywords if keyword.lower() in dest_prompt.lower())
        
        # If either has significant composition terms, consider it a composition change
        return source_count >= 2 or dest_count >= 2
    
    def _lighting_bridge_template(self, source_prompt, dest_prompt, analyses, lighting_weight, content_weight, composition_weight):
        """Template for lighting-focused bridges"""
        import random
        
        # Extract key subjects from prompts
        source_subject = self._extract_subject(source_prompt)
        dest_subject = self._extract_subject(dest_prompt)
        
        # Determine lighting descriptors
        lighting_terms = self._find_lighting_terms(source_prompt, dest_prompt)
        
        if not lighting_terms:
            lighting_terms = random.choice([
                "transitioning illumination",
                "changing light",
                "shifting shadows",
                "evolving atmosphere"
            ])
        
        # Create the bridge prompt
        bridge = f"{source_subject} with {lighting_terms}, gradually transforming into {dest_subject}"
        
        # Adjust with weights
        if lighting_weight < 0.5:
            bridge = bridge.replace(lighting_terms, "changing light")
        
        return bridge
    
    def _content_bridge_template(self, source_prompt, dest_prompt, analyses, lighting_weight, content_weight, composition_weight):
        """Template for content-focused bridges"""
        import random
        
        # Extract key subjects from prompts
        source_subject = self._extract_subject(source_prompt)
        dest_subject = self._extract_subject(dest_prompt)
        
        # Create transition verb
        transition_verb = self._select_transition_verb(source_subject, dest_subject)
        
        # Create the bridge prompt
        bridge = f"{source_subject} {transition_verb} {dest_subject}"
        
        # Add qualifiers based on weights
        if lighting_weight > 0.7:
            lighting = self._find_lighting_terms(source_prompt, dest_prompt)
            if lighting:
                bridge += f", with {lighting}"
        
        if composition_weight > 0.7:
            composition = self._find_composition_terms(source_prompt, dest_prompt)
            if composition:
                bridge += f", {composition}"
        
        return bridge
    
    def _composition_bridge_template(self, source_prompt, dest_prompt, analyses, lighting_weight, content_weight, composition_weight):
        """Template for composition-focused bridges"""
        import random
        
        # Extract key subjects from prompts
        source_subject = self._extract_subject(source_prompt)
        
        # Find composition terms
        composition_terms = self._find_composition_terms(source_prompt, dest_prompt)
        if not composition_terms:
            composition_terms = random.choice([
                "changing perspective",
                "shifting viewpoint",
                "transitioning frame",
                "evolving composition"
            ])
        
        # Create the bridge prompt
        bridge = f"{source_subject}, {composition_terms}"
        
        # Add content from destination if weight is high
        if content_weight > 0.7:
            dest_subject = self._extract_subject(dest_prompt)
            bridge += f", revealing {dest_subject}"
        
        return bridge
    
    def _extract_subject(self, prompt):
        """Extract the main subject from a prompt"""
        import re
        
        # Remove common style keywords
        style_keywords = ["high quality", "highly detailed", "masterpiece", "best quality", 
                        "intricate", "sharp focus", "clear", "4k", "8k", "uhd", "hdr"]
        
        cleaned = prompt
        for keyword in style_keywords:
            cleaned = cleaned.replace(keyword, "")
        
        # Get the first substantial chunk before a comma or other separator
        parts = re.split(r'[,;:]', cleaned)
        subject = parts[0].strip()
        
        # If too short, use more
        if len(subject) < 10 and len(parts) > 1:
            subject = (parts[0] + ", " + parts[1]).strip()
        
        return subject
    
    def _find_lighting_terms(self, source_prompt, dest_prompt):
        """Find lighting terms in the prompts"""
        import re
        
        lighting_patterns = [
            r'(bright|dark|dim|shadowy|illuminated|glowing|shining)\s+\w+',
            r'(sunset|sunrise|dawn|dusk|daylight|moonlight|backlit)',
            r'(warm|cool|soft|harsh)\s+light',
            r'(golden|blue|red)\s+hour'
        ]
        
        terms = []
        for pattern in lighting_patterns:
            for prompt in [source_prompt, dest_prompt]:
                matches = re.findall(pattern, prompt, re.IGNORECASE)
                if isinstance(matches, list):
                    terms.extend([m if isinstance(m, str) else m[0] for m in matches])
                else:
                    terms.extend([matches])
        
        # Remove duplicates and empty strings
        terms = [t for t in terms if t]
        terms = list(set(terms))
        
        if terms:
            return ", ".join(terms)
        else:
            return ""
    
    def _find_composition_terms(self, source_prompt, dest_prompt):
        """Find composition terms in the prompts"""
        import re
        
        composition_patterns = [
            r'(wide|close|medium)\s+(shot|angle|view)',
            r'(aerial|drone|bird\'s eye|worm\'s eye)\s+view',
            r'(looking|facing)\s+(up|down|away|toward)',
            r'(from|through)\s+\w+\s+(perspective|viewpoint)'
        ]
        
        terms = []
        for pattern in composition_patterns:
            for prompt in [source_prompt, dest_prompt]:
                matches = re.findall(pattern, prompt, re.IGNORECASE)
                if matches:
                    terms.extend([" ".join(m) for m in matches if m])
        
        # Remove duplicates and empty strings
        terms = [t for t in terms if t]
        terms = list(set(terms))
        
        if terms:
            return ", ".join(terms)
        else:
            return ""
    
    def _select_transition_verb(self, source_subject, dest_subject):
        """Select an appropriate transition verb"""
        import random
        
        # General transitions
        general_transitions = [
            "transitioning to",
            "gradually becoming",
            "morphing into",
            "transforming into",
            "evolving into",
            "shifting towards"
        ]
        
        # Specific transitions based on content types
        nature_keywords = ["forest", "mountain", "ocean", "lake", "river", "landscape", "sky", "beach"]
        urban_keywords = ["city", "street", "building", "architecture", "urban"]
        person_keywords = ["person", "man", "woman", "child", "face", "portrait"]
        
        # Check if source and dest are of specific types
        source_is_nature = any(keyword in source_subject.lower() for keyword in nature_keywords)
        source_is_urban = any(keyword in source_subject.lower() for keyword in urban_keywords)
        source_is_person = any(keyword in source_subject.lower() for keyword in person_keywords)
        
        dest_is_nature = any(keyword in dest_subject.lower() for keyword in nature_keywords)
        dest_is_urban = any(keyword in dest_subject.lower() for keyword in urban_keywords)
        dest_is_person = any(keyword in dest_subject.lower() for keyword in person_keywords)
        
        # Select appropriate transitions
        if source_is_nature and dest_is_urban:
            specific_transitions = [
                "giving way to",
                "opening up to reveal",
                "transitioning into"
            ]
        elif source_is_urban and dest_is_nature:
            specific_transitions = [
                "fading into",
                "being reclaimed by",
                "dissolving into"
            ]
        elif source_is_person and (dest_is_nature or dest_is_urban):
            specific_transitions = [
                "stepping into",
                "exploring",
                "entering"
            ]
        else:
            specific_transitions = []
        
        # Combine and select
        all_transitions = general_transitions + specific_transitions
        return random.choice(all_transitions)
    
    def _interpolate_embeddings(self, text_embeds, bridge_count, bridge_strength):
        """
        Create enhanced embeddings using interpolation between existing embeddings.
        
        Args:
            text_embeds: Original text embeddings
            bridge_count: Number of bridge embeddings per transition
            bridge_strength: How strongly to blend between embeddings
            
        Returns:
            Enhanced text embeddings
        """
        # Get the original embeddings
        original_embeds = text_embeds["prompt_embeds"]
        negative_embeds = text_embeds["negative_prompt_embeds"]
        
        # If only one embedding, nothing to interpolate
        if len(original_embeds) <= 1:
            return text_embeds
        
        # Create enhanced embeddings with interpolated bridges
        enhanced_embeds = []
        
        # Add each original with bridge embeddings after it
        for i in range(len(original_embeds) - 1):
            # Add the original embedding
            enhanced_embeds.append(original_embeds[i])
            
            # Add bridge embeddings
            src_embed = original_embeds[i]
            dst_embed = original_embeds[i + 1]
            
            for j in range(bridge_count):
                # Calculate interpolation factor
                factor = (j + 1) / (bridge_count + 1) * bridge_strength
                
                # Check if shapes match
                if src_embed.shape == dst_embed.shape:
                    # Linear interpolation
                    bridge_embed = src_embed * (1 - factor) + dst_embed * factor
                    enhanced_embeds.append(bridge_embed)
                else:
                    # Shapes don't match, duplicate the source
                    log.warning(f"Embedding shapes don't match at transition {i}: {src_embed.shape} vs {dst_embed.shape}")
                    enhanced_embeds.append(src_embed.clone())
        
        # Add the final original embedding
        enhanced_embeds.append(original_embeds[-1])
        
        # Create the result dictionary
        result = {
            "prompt_embeds": enhanced_embeds,
            "negative_prompt_embeds": negative_embeds,
        }
        
        return result
    
    def _encode_bridge_prompts(self, text_embeds, bridge_prompts, bridge_count, bridge_strength, t5_encoder):
        """
        Encode bridge prompts using the T5 encoder.
        
        Args:
            text_embeds: Original text embeddings
            bridge_prompts: Generated bridge prompts
            bridge_count: Number of bridge prompts per transition
            bridge_strength: How strongly to apply bridge modifications
            t5_encoder: T5 encoder for encoding prompts
            
        Returns:
            Enhanced text embeddings
        """
        import comfy.model_management as mm
        
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        
        # Get the original embeddings
        original_embeds = text_embeds["prompt_embeds"]
        negative_embeds = text_embeds["negative_prompt_embeds"]
        
        # If only one embedding, nothing to bridge
        if len(original_embeds) <= 1 or not bridge_prompts:
            return text_embeds
        
        # Move encoder to device
        encoder = t5_encoder["model"]
        dtype = t5_encoder["dtype"]
        encoder.model.to(device)
        
        # Create a list of all prompts
        encode_prompts = []
        for bridge in bridge_prompts:
            encode_prompts.append(bridge["prompt"])
        
        # Encode the bridge prompts
        bridge_embeds = []
        if encode_prompts:
            with torch.autocast(device_type=mm.get_autocast_device(device), dtype=dtype, enabled=True):
                bridge_embeds = encoder(encode_prompts, device)
        
        # Combine original and bridge embeddings
        enhanced_embeds = []
        
        # Build the enhanced embedding list
        for i in range(len(original_embeds) - 1):
            # Add original embedding
            enhanced_embeds.append(original_embeds[i])
            
            # Find bridges for this transition
            transition_bridges = [b for b in bridge_prompts if b["transition_idx"] == i]
            
            # Add the specified number of bridge embeddings
            for j in range(bridge_count):
                if j < len(transition_bridges):
                    # Find the corresponding bridge embedding
                    bridge_idx = encode_prompts.index(transition_bridges[j]["prompt"])
                    enhanced_embeds.append(bridge_embeds[bridge_idx])
                else:
                    # Fill in with interpolated bridge
                    src_embed = original_embeds[i]
                    dst_embed = original_embeds[i + 1]
                    
                    # Calculate interpolation factor
                    factor = (j + 1) / (bridge_count + 1) * bridge_strength
                    
                    # Check if shapes match
                    if src_embed.shape == dst_embed.shape:
                        # Linear interpolation
                        bridge_embed = src_embed * (1 - factor) + dst_embed * factor
                    else:
                        # Shapes don't match, duplicate the source
                        bridge_embed = src_embed.clone()
                        
                    enhanced_embeds.append(bridge_embed)
        
        # Add the final original embedding
        enhanced_embeds.append(original_embeds[-1])
        
        # Move encoder back to offload device
        encoder.model.to(offload_device)
        mm.soft_empty_cache()
        
        # Create the result dictionary
        result = {
            "prompt_embeds": enhanced_embeds,
            "negative_prompt_embeds": negative_embeds,
        }
        
        return result