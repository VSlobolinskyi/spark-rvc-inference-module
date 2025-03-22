import datetime
import time
import logging
import os
import platform
import shutil
import re
import numpy as np
import soundfile as sf
from pydub import AudioSegment
import torch
import threading
from queue import Queue, Empty
from contextlib import nullcontext

# Import modules from your packages
from spark.cli.SparkTTS import SparkTTS
from rvc_ui.initialization import vc

# Initialize the Spark TTS model (moved outside function to avoid reinitializing)
model_dir = "spark/pretrained_models/Spark-TTS-0.5B"
device = 0

def process_with_rvc(
    spk_item, input_path, vc_transform, f0method,
    file_index1, file_index2, index_rate, filter_radius,
    resample_sr, rms_mix_rate, protect, 
    output_path, cuda_stream=None
):
    """Process audio through RVC with a specific CUDA stream."""
    logging.info(f"Starting RVC inference for {input_path}...")
    
    # Set the CUDA stream if provided
    with torch.cuda.stream(cuda_stream) if cuda_stream and torch.cuda.is_available() else nullcontext():
        # Call RVC processing function
        f0_file = None  # We're not using an F0 curve file
        output_info, output_audio = vc.vc_single(
            spk_item, input_path, vc_transform, f0_file, f0method,
            file_index1, file_index2, index_rate, filter_radius,
            resample_sr, rms_mix_rate, protect
        )
    
    # Save RVC output (CPU operation)
    rvc_saved = False
    try:
        if isinstance(output_audio, str) and os.path.exists(output_audio):
            # Case 1: output_audio is a file path string
            shutil.copy2(output_audio, output_path)
            rvc_saved = True
        elif isinstance(output_audio, tuple) and len(output_audio) >= 2:
            # Case 2: output_audio might be (sample_rate, audio_data)
            sf.write(output_path, output_audio[1], output_audio[0])
            rvc_saved = True
        elif hasattr(output_audio, 'name') and os.path.exists(output_audio.name):
            # Case 3: output_audio might be a file-like object
            shutil.copy2(output_audio.name, output_path)
            rvc_saved = True
    except Exception as e:
        output_info += f"\nError saving RVC output: {str(e)}"
        
    logging.info(f"RVC inference completed for {input_path}")
    return rvc_saved, output_info


def split_into_sentences(text):
    """
    Split text into sentences using regular expressions.
    
    Args:
        text (str): The input text to split
        
    Returns:
        list: A list of sentences
    """
    # Split on period, exclamation mark, or question mark followed by space or end of string
    sentences = re.split(r'(?<=[.!?])\s+|(?<=[.!?])$', text)
    # Remove any empty sentences
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


def modified_get_vc(sid0_value, protect0_value, file_index2_component):
    """
    Modified function to get voice conversion parameters
    """
    protect1_value = protect0_value
    outputs = vc.get_vc(sid0_value, protect0_value, protect1_value)
    
    if isinstance(outputs, tuple) and len(outputs) >= 3:
        return outputs[0], outputs[1], outputs[3]
    
    return 0, protect0_value, file_index2_component.choices[0] if file_index2_component.choices else ""