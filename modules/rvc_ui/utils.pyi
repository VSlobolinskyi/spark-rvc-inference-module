# rvc_ui/utils.py
import os
import gradio as gr


# Function to update the audio path when a file is uploaded
def update_audio_path(uploaded_file):
    if uploaded_file is None:
        return ""
    if isinstance(uploaded_file, list):
        uploaded_file = uploaded_file[0]
    if isinstance(uploaded_file, dict):
        return uploaded_file.get("name", "")
    if hasattr(uploaded_file, "name"):
        return uploaded_file.name
    return str(uploaded_file)


# Function to lookup index files in a given directory
def lookup_indices(index_root, index_paths):
    for root, dirs, files in os.walk(index_root, topdown=False):
        for name in files:
            if name.endswith(".index") and "trained" not in name:
                index_paths.append(f"{root}/{name}")


# Function to refresh available model and index choices
def change_choices(weight_root, index_root):
    names = [name for name in os.listdir(weight_root) if name.endswith(".pth")]
    index_paths = []
    for root, dirs, files in os.walk(index_root, topdown=False):
        for name in files:
            if name.endswith(".index") and "trained" not in name:
                index_paths.append(f"{root}/{name}")
    return {"choices": sorted(names), "__type__": "update"}, {
        "choices": sorted(index_paths),
        "__type__": "update",
    }

from gradio.events import Dependency

# Custom Gradio ToolButton component
class ToolButton(gr.Button, gr.components.FormComponent):
    def __init__(self, **kwargs):
        super().__init__(variant="tool", **kwargs)

    def get_block_name(self):
        return "button"
    from typing import Callable, Literal, Sequence, Any, TYPE_CHECKING
    from gradio.blocks import Block
    if TYPE_CHECKING:
        from gradio.components import Timer


# Simple clean function to reset a field (used for GPU memory management)
def clean():
    return {"value": "", "__type__": "update"}


__all__ = [
    "update_audio_path",
    "lookup_indices",
    "change_choices",
    "ToolButton",
    "clean",
]