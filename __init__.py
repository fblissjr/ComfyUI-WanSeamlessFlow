# WanSeamlessFlow/__init__.py

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
import sys

# Find and patch WanVideoSampler
found = False
for module_name in sys.modules:
    if module_name.endswith("WanVideoWrapper.nodes") or module_name.endswith(
        "fork-ComfyUI-WanVideoWrapper.nodes"
    ):
        module = sys.modules[module_name]
        if hasattr(module, "WanVideoSampler"):
            # Patch the module
            from .integration import get_blend_function

            module.WanVideoSampler.get_blend_function = get_blend_function
            found = True
            print("✅ WanSeamlessFlow: Successfully integrated with WanVideoSampler")
            break

if not found:
    log.info(
        "⚠️ WanSeamlessFlow: Could not find WanVideoSampler module. Will try again later."
    )

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]