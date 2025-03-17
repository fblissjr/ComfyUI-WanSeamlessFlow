import torch
import comfy.model_management as mm
import gc
import sys
import os
import types

# Track if we've already patched
SAMPLER_PATCHED = False

# WanSeamlessFlow/integration.py


def get_blend_function():
    """
    Returns a blend function that can be used by WanVideoSampler.
    
    This function provides a stable API that WanVideoSampler can use
    regardless of internal implementation changes.
    """
    try:
        from .blending import blend_embeddings
        return blend_embeddings
    except (ImportError, Exception):
        # Provide a fallback implementation
        def fallback_blend(embed1, embed2, ratio, method="linear"):
            if embed1.shape != embed2.shape:
                print(f"⚠️ Cannot blend embeddings with shapes {embed1.shape} and {embed2.shape}")
                return embed1
                
            # Simple linear interpolation as fallback
            if method == "smooth":
                blend_ratio = ratio * ratio * (3 - 2 * ratio)
            elif method == "ease_in":
                blend_ratio = ratio * ratio
            elif method == "ease_out":
                blend_ratio = ratio * (2 - ratio)
            else:
                blend_ratio = ratio
                
            return embed1 * (1 - blend_ratio) + embed2 * blend_ratio
            
        return fallback_blend

# Example of usage in WanVideoSampler:
# from WanSeamlessFlow.integration import get_blend_function
# blend_fn = get_blend_function()
# positive = blend_fn(current_embed, next_embed, raw_ratio, method=blend_method)

def apply_blend(prompt_index, c, section_size, text_embeds, verbosity=0):
    """
    Apply blending between prompts at section boundaries with proper device handling.
    """
    # Check if blending is enabled
    blend_width = text_embeds.get("blend_width", 0)
    
    if blend_width > 0 and prompt_index < len(text_embeds["prompt_embeds"]) - 1:
        # Calculate position within section (0-1)
        position = (max(c) % section_size) / section_size
        # Calculate blend zone (as proportion of section)
        blend_zone = blend_width / section_size
        
        if position > (1.0 - blend_zone):
            # We're in the transition zone
            raw_ratio = (position - (1.0 - blend_zone)) / blend_zone
            
            # Apply the selected curve
            blend_method = text_embeds.get("blend_method", "linear")
            
            # Get embeddings - preserve their original device and dtype
            current_embed = text_embeds["prompt_embeds"][prompt_index]
            next_embed = text_embeds["prompt_embeds"][prompt_index + 1]
            
            # Note device and dtype for consistency checking
            if verbosity > 2:
                print(f"Blending prompts {prompt_index}→{prompt_index+1}, " 
                      f"ratio: {raw_ratio:.3f}, device: {current_embed.device}, dtype: {current_embed.dtype}")
            
            # Perform blending with proper curve application
            if blend_method == "smooth":
                blend_ratio = raw_ratio * raw_ratio * (3 - 2 * raw_ratio)
            elif blend_method == "ease_in":
                blend_ratio = raw_ratio * raw_ratio
            elif blend_method == "ease_out":
                blend_ratio = raw_ratio * (2 - raw_ratio)
            elif blend_method == "sine":
                import math
                blend_ratio = 0.5 - 0.5 * math.cos(raw_ratio * math.pi)
            else:
                blend_ratio = raw_ratio
                
            # Preserve dtype during blending
            original_dtype = current_embed.dtype
            result = current_embed * (1 - blend_ratio) + next_embed * blend_ratio
            return result.to(dtype=original_dtype)
    
    # Not in transition zone, return original embedding
    return text_embeds["prompt_embeds"][prompt_index]

def patch_wanvideo_sampler():
    """
    Patch the WanVideoSampler to enable seamless transitions.
    This is executed on module import.
    """
    global SAMPLER_PATCHED
    
    if SAMPLER_PATCHED:
        return
        
    # Find WanVideoWrapper module
    wan_module_path = None
    for module_path in sys.modules.keys():
        if module_path.endswith('WanVideoWrapper.nodes') or module_path.endswith('fork-ComfyUI-WanVideoWrapper.nodes'):
            wan_module_path = module_path
            break
    
    if not wan_module_path:
        print("⚠️ WanSeamlessFlow: Could not find WanVideoWrapper module. Integration disabled.")
        return
    
    # Get the module
    wan_module = sys.modules[wan_module_path]
    
    # Check if WanVideoSampler exists
    if not hasattr(wan_module, 'WanVideoSampler'):
        print("⚠️ WanSeamlessFlow: Could not find WanVideoSampler class. Integration disabled.")
        return
    
    # Get the original process method
    original_process = wan_module.WanVideoSampler.process
    
    # Define our patched process method
    def patched_process(self, *args, **kwargs):
        """
        Patched version of WanVideoSampler.process that supports seamless transitions.
        """
        # Extract the text_embeds from args or kwargs
        text_embeds = None
        if len(args) > 1:
            text_embeds = args[1]
        elif 'text_embeds' in kwargs:
            text_embeds = kwargs['text_embeds']
            
        # If no text_embeds or no blend_width, use original method
        if text_embeds is None or not isinstance(text_embeds, dict) or 'blend_width' not in text_embeds or text_embeds['blend_width'] <= 0:
            return original_process(self, *args, **kwargs)
            
        # We have text_embeds with blend_width, apply our patched logic
        verbosity = text_embeds.get('verbosity', 0)
        if verbosity > 0:
            print(f"WanSeamlessFlow: Processing with blend width {text_embeds['blend_width']}")
            
        # Add our apply_blend function to the self object so it's in scope
        self.apply_blend = apply_blend
        
        # Modify the original function code

        original_code = original_process.__code__
        
        # Find line with prompt selection
        new_code = []
        with open(original_process.__code__.co_filename, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                new_code.append(line)
                # Insert our blending logic after prompt selection
                if "prompt_index = min(int(max(c) / section_size), num_prompts - 1)" in line:
                    blend_code = (
                        "        # WanSeamlessFlow integration\n"
                        "        if hasattr(self, 'apply_blend') and 'blend_width' in text_embeds and text_embeds['blend_width'] > 0:\n"
                        "            positive = self.apply_blend(prompt_index, c, section_size, text_embeds, text_embeds.get('verbosity', 0))\n"
                        "        else:\n"
                        "            positive = text_embeds[\"prompt_embeds\"][prompt_index]\n"
                    )
                    new_code.append(blend_code)
                    # Skip the next line which would be the original prompt selection
                    if i+1 < len(lines) and "positive = text_embeds" in lines[i+1]:
                        new_code.pop()  # Remove the original line
                        
        # Write modified function to a temporary file
        temp_file = os.path.join(os.path.dirname(__file__), "temp_patched_process.py")
        with open(temp_file, 'w') as f:
            f.write("".join(new_code))
            
        # Execute the modified function
        local_vars = {}
        with open(temp_file, 'r') as f:
            exec(f.read(), globals(), local_vars)
            
        # Clean up
        os.remove(temp_file)
        
        # Apply the patch
        wan_module.WanVideoSampler.process = types.MethodType(patched_process, wan_module.WanVideoSampler)
        
        # Mark as patched
        SAMPLER_PATCHED = True
        
        print("✅ WanSeamlessFlow: Successfully integrated with WanVideoSampler")
        
    # Replace the process method with our patched version
    wan_module.WanVideoSampler.process = types.MethodType(patched_process, wan_module.WanVideoSampler)
    
    # Mark as patched
    SAMPLER_PATCHED = True
    print("✅ WanSeamlessFlow: Successfully patched WanVideoSampler")

# Execute patch on import
patch_wanvideo_sampler()